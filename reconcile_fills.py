"""Reconcile live_trades.db against the broker's OWN fill records.

WHY THIS EXISTS
---------------
live_trades.db was written from whatever price the caller had to hand at the
moment it recorded a close. Alpaca's submit_order returns BEFORE the fill
(filled_avg_price=None on essentially every order), so the recorder fell back
to the current bar close — a price that is near the fill but is not the fill.
alpaca.py now polls for the real fill before returning, so NEW trades are
recorded correctly, but every row written before that fix carries an
approximate price, and its pnl is wrong by the difference.

Measured on AlpacaLive 2026-08-14: 9 recorded round trips totalling +$0.2238
against +$0.0637 actually filled — a 3.5x overstatement on an account whose
whole balance is ~$53. On top of that one round trip (a DDOG position closed
outside the normal path) was never recorded at all, so a -$0.067 loss was
simply missing from the history.

The broker's FILL activity feed is authoritative: it is what actually happened
to the money. This rebuilds round trips from it and tells you exactly where the
local history disagrees.

SCOPE / LIMITS
--------------
- Alpaca only. It is the one broker here exposing a per-fill activity feed;
  IG/OANDA/IBKR have no equivalent, so their rows can't be checked this way.
- FIFO matching, long-only (buy opens, sell closes) — this project's engine
  convention. A short-selling account would need the mirror case added.
- Fees are NOT part of this. They never touch live_trades.db by design; they're
  surfaced separately (see BrokerFee / summarize_fees). This only fixes whether
  the recorded PRICES match the fills.

USAGE
    python reconcile_fills.py                    # report every account, change nothing
    python reconcile_fills.py --account AlpacaLive
    python reconcile_fills.py --apply            # back up the DB, then correct it
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import live_trades, position_attribution  # noqa: E402
from backtester.accounts import build_broker_accounts, list_accounts  # noqa: E402
from backtester.live_trades import UNATTRIBUTED_STRATEGY  # noqa: E402

import requests  # noqa: E402

# A recorded price this far from the real fill is treated as a genuine
# disagreement rather than float noise. Prices are dollars; sub-hundredth
# differences can't change a cent-rounded P&L on these quantities.
PRICE_TOLERANCE = 0.0001
# Widest gap allowed between a recorded exit and a real fill when deciding they
# describe the same event. Recording happens moments after the fill (the poll
# loop plus a DB write), so a few minutes is generous; anything further apart is
# a different trade and should be reported as missing, not silently matched.
MATCH_WINDOW_SECONDS = 600


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def fetch_fills(broker) -> list[dict]:
    """Every FILL activity for this account, oldest first.

    Paginated with page_token: the endpoint caps a page at 100, and an account
    that has been trading for a while will exceed that. Stopping at one page
    would silently reconcile against a partial history — worse than not
    reconciling at all, because it would report real trades as "missing".
    """
    base = "https://paper-api.alpaca.markets" if broker.is_paper else "https://api.alpaca.markets"
    headers = {
        "APCA-API-KEY-ID": broker._client._api_key,
        "APCA-API-SECRET-KEY": broker._client._secret_key,
    }
    fills: list[dict] = []
    page_token = None
    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        resp = requests.get(
            f"{base}/v2/account/activities/FILL", headers=headers, params=params, timeout=30
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        fills.extend(batch)
        if len(batch) < 100:
            break
        page_token = batch[-1].get("id")
        if not page_token:
            break
    return sorted(fills, key=lambda f: f["transaction_time"])


def aggregate_by_order(fills: list[dict]) -> list[dict]:
    """Collapse partial fills into one entry per ORDER, at the quantity-weighted
    average price — which is exactly what `filled_avg_price` reports and what
    live_trades.db stores one row of.

    A single market order routinely fills in several pieces (one 106-share PSKY
    order filled as 43 + 16 + 12 + 35 + 0.0211). Matching raw fills against the
    DB therefore reports one real round trip as several "missing" ones plus an
    "orphan" recorded row — which is what an earlier run of this tool did, and
    why it must never be trusted before this step.

    Ordered by the LAST fill of each order: that is the moment the position
    actually finished changing, and the timestamp the recorder used.
    """
    orders: dict[str, dict] = {}
    for fill in fills:
        # Fall back to a per-fill identity if the broker ever omits order_id,
        # so an unknown order can't silently merge with an unrelated one.
        key = fill.get("order_id") or f"{fill.get('id')}"
        qty, price = float(fill["qty"]), float(fill["price"])
        order = orders.get(key)
        if order is None:
            orders[key] = {
                "symbol": fill["symbol"], "side": fill["side"],
                "qty": qty, "notional": qty * price,
                "transaction_time": fill["transaction_time"],
            }
            continue
        order["qty"] += qty
        order["notional"] += qty * price
        order["transaction_time"] = max(order["transaction_time"], fill["transaction_time"])

    merged = [
        {
            "symbol": o["symbol"], "side": o["side"], "qty": str(o["qty"]),
            "price": str(o["notional"] / o["qty"] if o["qty"] else 0.0),
            "transaction_time": o["transaction_time"],
        }
        for o in orders.values()
    ]
    return sorted(merged, key=lambda f: f["transaction_time"])


def build_round_trips(fills: list[dict]) -> tuple[list[dict], list[dict]]:
    """FIFO-match fills into closed round trips. Returns (round_trips, still_open).

    Partial closes are handled: a sell smaller than the open lot closes part of
    it and leaves the remainder open; a sell larger than one lot consumes
    several. Both happen here because sizing is a percentage of equity, so lot
    sizes drift between entries.
    """
    lots: dict[str, deque] = defaultdict(deque)
    round_trips: list[dict] = []

    for fill in fills:
        symbol = fill["symbol"]
        qty = float(fill["qty"])
        price = float(fill["price"])
        when = fill["transaction_time"]

        if fill["side"] == "buy":
            lots[symbol].append({"qty": qty, "price": price, "time": when})
            continue

        remaining = qty
        while remaining > 1e-12 and lots[symbol]:
            lot = lots[symbol][0]
            matched = min(remaining, lot["qty"])
            round_trips.append({
                "ticker": symbol,
                "qty": matched,
                "entry_time": lot["time"],
                "entry_price": lot["price"],
                "exit_time": when,
                "exit_price": price,
                "pnl": (price - lot["price"]) * matched,
            })
            lot["qty"] -= matched
            remaining -= matched
            if lot["qty"] <= 1e-12:
                lots[symbol].popleft()
        # remaining > 0 here would mean selling more than was ever bought —
        # only possible on a short, which this project's engine doesn't do on
        # Alpaca. Left unhandled deliberately rather than silently absorbed.

    still_open = [
        {"ticker": symbol, **lot} for symbol, dq in lots.items() for lot in dq if lot["qty"] > 1e-12
    ]
    return round_trips, still_open


def find_superseded(conn, account_id: str, orphans: list[dict]) -> list[dict]:
    """Orphan rows that are a SUPERSEDED AGGREGATE of rows now in the table.

    A position opened once can be closed in several sells on different days.
    The recorder wrote that as ONE row for the whole quantity; the broker's
    fills show it as separate round trips, which get inserted individually.
    Both then exist and the P&L is counted twice — the exact residual seen on
    MyAlpaca (+$2.6225) after the first --apply run.

    Deliberately strict, because this DELETES rows: the pieces must share the
    orphan's ticker AND entry price AND entry time, must not include the orphan
    itself, and must sum to its quantity. Anything short of that is reported for
    a human to look at, never removed.
    """
    superseded = []
    for orphan in orphans:
        pieces = [
            dict(r)
            for r in conn.execute(
                """SELECT * FROM live_trades
                   WHERE account_id = ? AND ticker = ? AND id != ?
                     AND abs(entry_price - ?) < ? AND substr(entry_time, 1, 16) = substr(?, 1, 16)""",
                (
                    account_id, orphan["ticker"], orphan["id"],
                    orphan["entry_price"], PRICE_TOLERANCE, orphan["entry_time"],
                ),
            )
        ]
        if not pieces:
            continue
        if abs(sum(p["qty"] for p in pieces) - orphan["qty"]) < 1e-6:
            superseded.append({"orphan": orphan, "pieces": pieces})
    return superseded


def reconcile_account(account: dict, apply: bool, drop_superseded: bool = False) -> dict:
    """Compare one account's recorded history to its real fills."""
    broker = build_broker_accounts([account["id"]])[0]
    fills = fetch_fills(broker)
    # Aggregate BEFORE matching — see aggregate_by_order for why skipping this
    # produces a confidently wrong report.
    truth, still_open = build_round_trips(aggregate_by_order(fills))

    conn = sqlite3.connect(live_trades.DB_PATH)
    conn.row_factory = sqlite3.Row
    recorded = [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM live_trades WHERE account_id = ? ORDER BY exit_time", (account["id"],)
        )
    ]

    unmatched = list(recorded)
    corrections: list[dict] = []
    missing: list[dict] = []

    for actual in truth:
        exit_at = _parse_time(actual["exit_time"])
        # Same ticker, same size, closed at about the same moment. Quantity is
        # part of the key because a ticker can be round-tripped twice in one
        # session (DDOG was, on 2026-08-11).
        candidates = [
            row for row in unmatched
            if row["ticker"] == actual["ticker"]
            and abs(row["qty"] - actual["qty"]) < 1e-6
            and (_parse_time(row["exit_time"]) is not None and exit_at is not None)
            and abs((_parse_time(row["exit_time"]) - exit_at).total_seconds()) <= MATCH_WINDOW_SECONDS
        ]
        if not candidates:
            missing.append(actual)
            continue
        row = min(
            candidates,
            key=lambda r: abs((_parse_time(r["exit_time"]) - exit_at).total_seconds()),
        )
        unmatched.remove(row)
        if (
            abs(row["entry_price"] - actual["entry_price"]) > PRICE_TOLERANCE
            or abs(row["exit_price"] - actual["exit_price"]) > PRICE_TOLERANCE
        ):
            corrections.append({"row": row, "actual": actual})

    result = {
        "nickname": account["nickname"],
        "fills": len(fills),
        "recorded": recorded,
        "recorded_total": sum(r["pnl"] for r in recorded),
        "actual_total": sum(t["pnl"] for t in truth),
        "corrections": corrections,
        "missing": missing,
        "orphans": unmatched,   # in the DB but no matching fill
        "still_open": still_open,
    }

    if apply and (corrections or missing):
        _apply(conn, account, corrections, missing)

    # Computed AFTER any inserts above, since an orphan only becomes provably
    # superseded once its replacement pieces are actually in the table.
    result["superseded"] = find_superseded(conn, account["id"], unmatched)
    if drop_superseded and result["superseded"]:
        conn.executemany(
            "DELETE FROM live_trades WHERE id = ?",
            [(s["orphan"]["id"],) for s in result["superseded"]],
        )
        conn.commit()
    conn.close()
    return result


def _apply(conn, account: dict, corrections: list[dict], missing: list[dict]) -> None:
    """Rewrite wrong prices and insert round trips that were never recorded."""
    for c in corrections:
        row, actual = c["row"], c["actual"]
        pnl = (actual["exit_price"] - actual["entry_price"]) * actual["qty"]
        pnl_pct = (
            (actual["exit_price"] - actual["entry_price"]) / actual["entry_price"]
            if actual["entry_price"] else 0.0
        )
        conn.execute(
            "UPDATE live_trades SET entry_price = ?, exit_price = ?, pnl = ?, pnl_pct = ? WHERE id = ?",
            (actual["entry_price"], actual["exit_price"], pnl, pnl_pct, row["id"]),
        )

    for actual in missing:
        # Attribution may still exist if the position was opened normally and
        # only its CLOSE went unrecorded; fall back to the explicit
        # "(unattributed)" marker rather than guessing a strategy, so these
        # never quietly inflate a real strategy's measured performance.
        # Read-only peek (load_map, not pop_open): a live position of the same
        # ticker may legitimately still be open and own that attribution.
        strategy = UNATTRIBUTED_STRATEGY
        try:
            entry = position_attribution.load_map().get(
                f"{account['id']}|{actual['ticker']}"
            )
            if entry and entry.get("strategy_name"):
                strategy = entry["strategy_name"]
        except Exception:  # noqa: BLE001 — attribution is a nicety, not a blocker
            pass
        pnl = (actual["exit_price"] - actual["entry_price"]) * actual["qty"]
        pnl_pct = (
            (actual["exit_price"] - actual["entry_price"]) / actual["entry_price"]
            if actual["entry_price"] else 0.0
        )
        conn.execute(
            """INSERT INTO live_trades
               (account_id, ticker, strategy_name, is_paper, entry_time, entry_price,
                exit_time, exit_price, qty, pnl, pnl_pct, conviction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                account["id"], actual["ticker"], strategy, int(account["is_paper"]),
                actual["entry_time"], actual["entry_price"],
                actual["exit_time"], actual["exit_price"],
                actual["qty"], pnl, pnl_pct, None,
            ),
        )
    conn.commit()


def _report(result: dict) -> None:
    print(f"\n{'=' * 78}\n{result['nickname']}  —  {result['fills']} fills from the broker\n{'=' * 78}")
    print(f"  recorded in live_trades.db : {result['recorded_total']:+.4f}  "
          f"({len(result['recorded'])} round trips)")
    print(f"  actually filled            : {result['actual_total']:+.4f}")
    delta = result["recorded_total"] - result["actual_total"]
    print(f"  overstatement              : {delta:+.4f}")

    if result["corrections"]:
        print(f"\n  {len(result['corrections'])} rows with prices that don't match the fill:")
        for c in result["corrections"]:
            row, actual = c["row"], c["actual"]
            print(f"    id={row['id']:<4} {row['ticker']:<6} "
                  f"in {row['entry_price']:>9.4f}->{actual['entry_price']:>9.4f}  "
                  f"out {row['exit_price']:>9.4f}->{actual['exit_price']:>9.4f}   "
                  f"pnl {row['pnl']:>+8.4f} -> {actual['pnl']:>+8.4f}")

    if result["missing"]:
        print(f"\n  {len(result['missing'])} round trips that HAPPENED but were never recorded:")
        for m in result["missing"]:
            print(f"    {m['ticker']:<6} qty={m['qty']:.4f}  "
                  f"{str(m['entry_time'])[:19]} @ {m['entry_price']:.4f}  ->  "
                  f"{str(m['exit_time'])[:19]} @ {m['exit_price']:.4f}   pnl {m['pnl']:+.4f}")

    superseded_ids = {s["orphan"]["id"] for s in result.get("superseded", [])}
    if result.get("superseded"):
        print(f"\n  {len(result['superseded'])} DOUBLE-COUNTED rows — an old single-row close "
              f"whose real pieces are now recorded separately:")
        for s in result["superseded"]:
            o = s["orphan"]
            print(f"    id={o['id']:<4} {o['ticker']:<6} qty={o['qty']:.4f} pnl={o['pnl']:+.4f}"
                  f"   superseded by ids {[p['id'] for p in s['pieces']]} "
                  f"(qty {sum(p['qty'] for p in s['pieces']):.4f}, "
                  f"pnl {sum(p['pnl'] for p in s['pieces']):+.4f})")

    remaining_orphans = [o for o in result["orphans"] if o["id"] not in superseded_ids]
    if remaining_orphans:
        print(f"\n  {len(remaining_orphans)} recorded rows with NO matching fill (investigate, "
              f"not auto-corrected):")
        for o in remaining_orphans:
            print(f"    id={o['id']:<4} {o['ticker']:<6} qty={o['qty']:.4f} "
                  f"exit={str(o['exit_time'])[:19]} pnl={o['pnl']:+.4f}")

    if result["still_open"]:
        print("\n  currently open (not a discrepancy, shown for completeness):")
        for s in result["still_open"]:
            print(f"    {s['ticker']:<6} qty={s['qty']:.4f} @ {s['price']:.4f}")

    if not (result["corrections"] or result["missing"] or result["orphans"]
            or result.get("superseded")):
        print("\n  Local history matches the broker exactly. Nothing to correct.")


def main(argv: list[str]) -> int:
    apply = "--apply" in argv
    drop_superseded = "--drop-superseded" in argv
    wanted = None
    for i, arg in enumerate(argv):
        if arg == "--account" and i + 1 < len(argv):
            wanted = argv[i + 1]
        elif arg.startswith("--account="):
            wanted = arg.split("=", 1)[1]

    accounts = [
        a for a in list_accounts()
        if a.get("broker") == "alpaca" and (wanted is None or a["nickname"] == wanted)
    ]
    if not accounts:
        print("No matching Alpaca accounts. (This tool is Alpaca-only — see module docstring.)")
        return 1

    if apply or drop_superseded:
        backup = live_trades.DB_PATH.with_suffix(
            f".db.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        shutil.copy2(live_trades.DB_PATH, backup)
        print(f"Backed up live_trades.db -> {backup.name}")

    changed = False
    for account in accounts:
        try:
            result = reconcile_account(account, apply, drop_superseded)
        except Exception as e:  # noqa: BLE001 — one unreachable account shouldn't stop the rest
            print(f"\n{account['nickname']}: could not reconcile — {e}")
            continue
        _report(result)
        changed = changed or bool(
            result["corrections"] or result["missing"] or result.get("superseded")
        )

    if changed and not (apply or drop_superseded):
        print("\nThis was a REPORT ONLY — nothing was changed.")
        print("Re-run with --apply to correct prices and insert missing round trips,")
        print("and/or --drop-superseded to remove provably double-counted rows.")
        print("Either flag backs the database up first.")
    elif apply or drop_superseded:
        print("\nDatabase corrected. Restart the dashboard and mobile backend so they "
              "re-read it:\n  python restart_all.py --only dashboard,mobile_backend")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
