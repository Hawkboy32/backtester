"""Part 3 of the confirmation-entry investigation (see plan
woolly-zooming-kurzweil.md): before spending effort building or tuning a
"wait for the turn" entry filter, check whether real losing trades would
ever have BENEFITED from one — did price ever show a recovery attempt before
the eventual losing exit, or was the slide to the exit monotonic from the
start?

Read-only against live_trades.db and Polygon's historical minute bars (this
is closed history, exactly the intended backtesting use of Polygon under
this project's "live-only sources for live signals" policy - nothing here
touches a live path).

SCOPE: every live (is_paper=0) VWAP Mean Reversion / Bollinger Mean
Reversion trade ever recorded, not just today's active roster - retired
tickers' history is exactly the data this question needs, and these are the
only two strategies with any live rows at all (crypto/forex extra_targets
haven't closed a live trade yet, confirmed via a direct query before writing
this). Both strategies are long-only as coded (only ever emit BUY to open),
so no short-side handling is needed here.

Run:  python diagnose_turn_before_loss.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from backtester.data import PolygonClient  # noqa: E402
from backtester.live_trades import DB_PATH  # noqa: E402

import sqlite3  # noqa: E402

# Same grouping definition live_trades._group_into_decisions uses (10s,
# anchored on the group's first row) - reimplemented here rather than
# imported because the shared helper's tuple shape doesn't carry entry_time/
# entry_price/account_id through, and this diagnostic needs all of them.
DECISION_GROUP_SECONDS = 10.0

# How far before the recorded entry_time to pull bars, to see the setup
# forming - not used for the turn/no-turn verdict itself (that only looks at
# entry->exit), just gives the printed excerpt a moment of context.
LEAD_IN_MINUTES = 5

# A "tick up" requires this many consecutive higher closes to count as a
# genuine uptick rather than single-bar noise - deliberately the SAME shape
# as the acceptance_bars/confirm_turn_bars edge-detection elsewhere in this
# codebase, not a new threshold invented just for this script.
MIN_CONSECUTIVE_UP_CLOSES = 2

STRATEGIES = ("VWAP Mean Reversion", "Bollinger Mean Reversion")


def fetch_live_losing_decisions() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    placeholders = ",".join("?" for _ in STRATEGIES)
    rows = con.execute(
        f"""SELECT ticker, strategy_name, account_id, entry_time, entry_price,
                    exit_time, exit_price, pnl, pnl_pct
             FROM live_trades
             WHERE is_paper = 0 AND strategy_name IN ({placeholders})
             ORDER BY ticker, strategy_name, exit_time""",
        STRATEGIES,
    ).fetchall()

    decisions: list[dict] = []
    by_combo: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for r in rows:
        by_combo.setdefault((r["ticker"], r["strategy_name"]), []).append(r)

    def flush(ticker: str, strategy_name: str, group: list[sqlite3.Row]) -> None:
        # Takes the current group as a REQUIRED explicit argument, not a
        # default - a default is evaluated once at function-definition time,
        # which is exactly the bug this replaced: every call after the first
        # gap kept re-emitting the FIRST group's stale data instead of the
        # current one, because a closure default doesn't track reassignment
        # of the outer variable (confirmed live: produced 7 identical MPWR
        # rows and 4 identical PKG rows before this fix).
        if not group:
            return
        pcts = [r["pnl_pct"] for r in group]
        mean_pct = sum(pcts) / len(pcts)
        entry_times = [datetime.fromisoformat(r["entry_time"].replace("Z", "+00:00")) for r in group]
        exit_times = [datetime.fromisoformat(r["exit_time"].replace("Z", "+00:00")) for r in group]
        decisions.append({
            "ticker": ticker,
            "strategy_name": strategy_name,
            "accounts": sorted({r["account_id"] for r in group}),
            "entry_time": min(entry_times),
            "exit_time": max(exit_times),
            "entry_price": sum(r["entry_price"] for r in group) / len(group),
            "exit_price": sum(r["exit_price"] for r in group) / len(group),
            "pnl": sum(r["pnl"] for r in group),
            "pnl_pct": mean_pct,
            "is_loss": mean_pct < 0,
        })

    for (ticker, strategy_name), combo_rows in by_combo.items():
        group: list[sqlite3.Row] = []
        anchor: datetime | None = None
        for r in combo_rows:
            ts = datetime.fromisoformat(r["exit_time"].replace("Z", "+00:00"))
            if anchor is not None and (ts - anchor).total_seconds() > DECISION_GROUP_SECONDS:
                flush(ticker, strategy_name, group)
                group = []
                anchor = None
            group.append(r)
            anchor = anchor or ts
        flush(ticker, strategy_name, group)

    return [d for d in decisions if d["is_loss"]]


def analyze_decision(client: PolygonClient, d: dict) -> dict:
    from_dt = d["entry_time"] - timedelta(minutes=LEAD_IN_MINUTES)
    to_dt = d["exit_time"] + timedelta(minutes=1)
    bars = client.get_aggregates(
        ticker=d["ticker"],
        from_date=from_dt.date().isoformat(),
        to_date=to_dt.date().isoformat(),
        multiplier=1,
        timespan="minute",
    )
    if bars.empty:
        return {**d, "bars_found": 0}

    window = bars[(bars.index >= d["entry_time"]) & (bars.index <= d["exit_time"])]
    if window.empty:
        return {**d, "bars_found": 0}

    entry_price = d["entry_price"]
    closes = window["close"]
    lows = window["low"]

    adverse_pct = ((lows - entry_price) / entry_price * 100).min()

    # Did price ever tick up MIN_CONSECUTIVE_UP_CLOSES bars in a row at any
    # point during the hold (a candidate "turn"), regardless of what
    # happened after? Same up-run edge-detection shape as acceptance_bars.
    up = closes.diff() > 0
    ever_upticked = bool(up.rolling(MIN_CONSECUTIVE_UP_CLOSES).sum().eq(MIN_CONSECUTIVE_UP_CLOSES).any())

    # A stronger, unambiguous signal: did price ever close back at/above the
    # ENTRY price at any point before the eventual losing exit - a genuine
    # recovery attempt, not just a two-bar wiggle.
    at_or_above = closes >= entry_price
    ever_recovered_to_entry = bool(at_or_above.any())

    # Decompose a "yes" two ways, since they imply very different fixes:
    # exiting SOON after the last recovery (a real reversion the strategy's
    # own logic likely caught, that still realized a small loss - consistent
    # with slippage/multi-account fill lag eating a thin margin, which
    # confirmation-entry does nothing about) vs a LONG gap (price recovered,
    # then genuinely fell away again on a real second leg down - the case
    # confirmation-entry is actually trying to address).
    bars_since_recovery = None
    if ever_recovered_to_entry:
        last_recovery_pos = at_or_above[at_or_above].index[-1]
        bars_since_recovery = len(window[window.index > last_recovery_pos])

    return {
        **d,
        "bars_found": len(window),
        "max_adverse_pct": adverse_pct,
        "ever_upticked": ever_upticked,
        "ever_recovered_to_entry": ever_recovered_to_entry,
        "bars_since_recovery": bars_since_recovery,
    }


def main() -> int:
    losing = fetch_live_losing_decisions()
    print(f"Live losing decisions found (VWAP + Bollinger Mean Reversion, all-time): {len(losing)}\n")
    if not losing:
        print("Nothing to analyze.")
        return 0

    client = PolygonClient()
    results = []
    for i, d in enumerate(sorted(losing, key=lambda x: x["exit_time"])):
        try:
            r = analyze_decision(client, d)
        except Exception as e:  # noqa: BLE001
            r = {**d, "bars_found": 0, "error": str(e)}
        results.append(r)
        # This account's Polygon plan has a real ~5 req/min free-tier ceiling
        # (documented project-wide, e.g. signal_service.py) - 0.15s between
        # requests hit that immediately and cost 3 of 15 decisions to 429s on
        # the first run. 13s keeps every request comfortably under 5/min.
        time.sleep(13.0)

    print(f"{'ticker':7s} {'strategy':24s} {'exit':17s} {'pnl%':>7s} {'bars':>5s} "
          f"{'max_adv%':>9s} {'upticked':>9s} {'recov>=entry':>13s} {'bars_since':>11s}")
    no_data = 0
    for r in results:
        if r.get("bars_found", 0) == 0:
            no_data += 1
            print(f"{r['ticker']:7s} {r['strategy_name']:24s} "
                  f"{r['exit_time'].strftime('%Y-%m-%d %H:%M'):17s} {r['pnl_pct']*100:>6.2f}% "
                  f"{'--':>5s}  (no bar data{': ' + r['error'] if r.get('error') else ''})")
            continue
        bsr = r.get("bars_since_recovery")
        print(f"{r['ticker']:7s} {r['strategy_name']:24s} "
              f"{r['exit_time'].strftime('%Y-%m-%d %H:%M'):17s} {r['pnl_pct']*100:>6.2f}% "
              f"{r['bars_found']:>5d} {r['max_adverse_pct']:>8.2f}% "
              f"{'yes' if r['ever_upticked'] else 'no':>9s} "
              f"{'yes' if r['ever_recovered_to_entry'] else 'no':>13s} "
              f"{(str(bsr) if bsr is not None else '-'):>11s}")

    scored = [r for r in results if r.get("bars_found", 0) > 0]
    n = len(scored)
    print(f"\n{'=' * 78}")
    print("SUMMARY")
    print(f"{'=' * 78}")
    print(f"  losing decisions analyzed: {n} (of {len(losing)}; {no_data} had no bar data)")
    if n == 0:
        return 0
    upticked = sum(1 for r in scored if r["ever_upticked"])
    recovered = sum(1 for r in scored if r["ever_recovered_to_entry"])
    print(f"  showed ANY {MIN_CONSECUTIVE_UP_CLOSES}-bar uptick before the losing exit: {upticked}/{n} ({upticked/n*100:.0f}%)")
    print(f"  never upticked at all (monotonic slide to exit):    {n - upticked}/{n} ({(n-upticked)/n*100:.0f}%)")
    print(f"  recovered to/above entry price before the exit:     {recovered}/{n} ({recovered/n*100:.0f}%)")
    print(f"  never recovered to entry price at all:               {n - recovered}/{n} ({(n-recovered)/n*100:.0f}%)")

    recovered_rows = [r for r in scored if r["ever_recovered_to_entry"]]
    if recovered_rows:
        near_miss = sum(1 for r in recovered_rows if (r["bars_since_recovery"] or 0) <= 3)
        second_leg = len(recovered_rows) - near_miss
        print()
        print(f"  of those {len(recovered_rows)} 'recovered to entry' cases, decomposed by what happened AFTER:")
        print(f"    exited within 3 bars of the last recovery (near-miss / likely slippage-eaten): "
              f"{near_miss}/{len(recovered_rows)}")
        print(f"    fell away again for a real second leg before the eventual exit:                "
              f"{second_leg}/{len(recovered_rows)}")
    print()
    print("READING THIS (both directions reported honestly, not just the favorable one):")
    print(f"  - If most losses NEVER uptick at all, that's evidence AGAINST confirmation-entry -")
    print(f"    there was nothing to wait for, the move was simply wrong from the first tick.")
    print(f"  - If most losses DO uptick but still end up losing, that's ALSO evidence against it -")
    print(f"    waiting for that same uptick would have entered into the same eventual failure.")
    print(f"  - A meaningful 'recovered to entry' rate is the more useful signal: it means price came")
    print(f"    back far enough that a turn-confirmed entry might have exited near breakeven instead")
    print(f"    of never entering, or entering into the eventual loss anyway - worth the Part 1 predicate")
    print(f"    replay (not yet built) to say which, precisely.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
