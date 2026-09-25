"""Actually EXECUTES copy-trading signals — poll each active roster investor
(copy_trading.poll_investor), and for every opened/closed position, place
the equivalent order on Chopper's own EXISTING paper broker accounts,
through the exact same execution/logging/notification path auto_trader.py
itself uses (execute_order_across_accounts, execution_log.log_results,
notifications.notify_trade_open/close/rejected) — reused, not reimplemented,
so this inherits the same audit trail and phone notifications every other
strategy's trades already get. Every copy-trade is tagged "Copy: <username>"
so it's unmistakable in the execution log and on your phone.

PAPER ACCOUNTS BY DEFAULT, LIVE FOR A NAMED ALLOWLIST — see
LIVE_COPY_USERNAMES below. Started paper-only (2026-09-16 agreed plan: prove
out on Demo/paper first); extended live 2026-09-23, deliberately, ONLY for
the two investors (RainbirdFx, Aukie2008) who'd actually produced a real
paper track record by then — the other two linked investors (campervans,
celesh) hadn't produced a single trade yet, so there was nothing to judge
them on. Live-eligible investors still ALSO copy to every matching paper
account, same as before — this adds a target, it doesn't replace paper
validation.

LONG OPENS ONLY for v1 — an eToro short (isBuy=False) is logged and
skipped, not executed, since the target paper accounts default to
long_only (see accounts.py's account_position_modes) and this hasn't been
tested against a short-enabled account yet.

Sizing: {COPY_SIZING_VALUE}% of each target account's own equity per copied
position (SizingMode.PCT_EQUITY), using the investor's own eToro openRate as
the sizing reference price — same "reference price is a sizing hint only,
never the actual fill price" contract every other caller of
compute_qty_for_account already relies on. A conservative default, easy to
change (see the constant below) — not a tuned/optimized value.

Run in a terminal (foreground, Ctrl+C to stop), same as copy_trader.py — not
yet managed by restart_all.py or any watchdog. Run copy_trader.py (observe
only) and this script are mutually exclusive on the same investor: running
both would double-process the same events against two independent state
files (copy_trading.py's per-username JSON is shared by both, so actually
they'd interfere — copy_trader.py should be STOPPED before running this).

Usage:
    python copy_trade_executor.py --username Aukie2008 --username RainbirdFx --interval 90
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import keyring

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import live_trades, notifications  # noqa: E402
from backtester.accounts import (  # noqa: E402
    KEYRING_SERVICE,
    _keyring_key,
    _load_raw,
    account_asset_class,
    build_broker_accounts,
    infer_asset_class,
    list_accounts,
)
from backtester.copy_roster import (  # noqa: E402
    CopyOrderFill,
    OpenCopyPosition,
    active_usernames,
    ensure_entries,
    find_open_position,
    load_roster,
    remove_open_position,
    save_roster,
)
from backtester.copy_trading import CopyEvent, CopyTradingError, poll_investor  # noqa: E402
from backtester.execution import (  # noqa: E402
    AccountOrder,
    SizingMode,
    compute_qty_for_account,
    execute_order_across_accounts,
)
from backtester.execution_log import log_results  # noqa: E402
from backtester.brokers.base import OrderSide  # noqa: E402

COPY_SIZING_MODE = SizingMode.PCT_EQUITY
# % of each target account's own equity, per copied position — same rule
# live and paper. Raised from 2.0 to 98.0 on 2026-09-23, deliberately, the
# day live copy-trading (LIVE_COPY_USERNAMES) started: 2% was calibrated
# against MyAlpaca's unrealistic $100k paper balance (~$2000/position,
# looked meaningful but proved nothing about how sizing behaves on a real,
# small account) and the user explicitly wanted to see real behaviour on a
# small live balance instead, not a diluted-down version of the paper test.
# On AlpacaLive's actual ~$50 equity this means each position uses ~98% of
# the account - if an investor opens multiple positions in one poll cycle,
# only the first gets funded; the rest fail sizing/buying-power cleanly
# (no phantom fills, no corrupted state - same guards as everywhere else in
# this file) rather than being copied partially.
COPY_SIZING_VALUE = 98.0

# Investors whose copy-trades also execute on a LIVE account, not just paper
# — decided 2026-09-23 after reviewing paper performance: these two are the
# only ones that had actually produced trades to judge (8 total, 100% win,
# but n=8 — explicitly a thin sample, not a proven track record the way
# VWAP Mean Reversion's 117-trade history is). campervans and celesh stay
# paper-only until they show something real. Revisit this set as more paper
# evidence comes in — it's a judgment call snapshot, not a permanent rule.
LIVE_COPY_USERNAMES = {"RainbirdFx", "Aukie2008"}

# The one live account copy-trading is allowed to touch. AlpacaLive
# specifically because it's the only live account currently trading
# equities — what these investors actually hold (MU, GOOG, ARM, META, ...).
# IBKR Live runs the FX strategies; not a fit for copying equity positions.
LIVE_COPY_ACCOUNT_NICKNAME = "AlpacaLive"

DEFAULT_INTERVAL_SECONDS = 90


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _find_etoro_credentials(account_nickname: str | None) -> tuple[str, str, str]:
    candidates = [a for a in _load_raw() if a["broker"] == "etoro"]
    if account_nickname:
        candidates = [a for a in candidates if a["nickname"] == account_nickname]
    if not candidates:
        raise SystemExit("No linked eToro account found. Link one via the Accounts tab first.")
    if len(candidates) > 1:
        names = ", ".join(a["nickname"] for a in candidates)
        raise SystemExit(f"Multiple eToro accounts linked ({names}) — pass --account-nickname to pick one.")
    account = candidates[0]
    api_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "api_key"))
    user_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "secret_key"))
    if not api_key or not user_key:
        raise SystemExit(f"Credentials for '{account['nickname']}' are missing from the OS keyring.")
    return account["nickname"], api_key, user_key


def _target_accounts(username: str, ticker: str) -> list[dict]:
    """Linked accounts to copy this investor's event onto: every matching
    PAPER account (never an eToro account — this routes through Chopper's
    OWN brokers, per the 2026-09-16 decision), PLUS the live account too if
    this username is in LIVE_COPY_USERNAMES (added 2026-09-23, see that
    constant's own docstring for why only two investors are live-eligible).

    KNOWN GAP, confirmed live 2026-09-21: IBKR Paper is one of the paper
    targets, but IBKR does not allow a live and paper Gateway session to
    stay logged in concurrently under the same login — confirmed by
    directly testing (stopped IBKR Live, IBKR Paper immediately logged in
    and worked fully; restarting Live logged Paper back out). This isn't a
    settings/port issue — a second IBC instance with its own
    TWS_SETTINGS_PATH was tried first and didn't help, because the
    conflict is server-side (IBKR's own session handling), not local.
    Practical effect: since IBKR Live needs to stay up for real trading,
    IBKR Paper as a copy-trade target will reliably fail with a
    connection-refused sizing error every time this runs, by design of the
    current single-login setup — not a bug in this function, see
    ibgateway_watchdog.py for why it isn't monitored either. Only real fix:
    a genuinely separate paper-trading login, if IBKR issues one for this
    account (unconfirmed) - checking that is still open."""
    asset_class = infer_asset_class(ticker)
    targets = [
        a for a in list_accounts()
        if a["is_paper"] and a["broker"] != "etoro" and account_asset_class(a) == asset_class
    ]
    if username in LIVE_COPY_USERNAMES:
        targets += [
            a for a in list_accounts()
            if not a["is_paper"]
            and a["nickname"] == LIVE_COPY_ACCOUNT_NICKNAME
            and account_asset_class(a) == asset_class
        ]
    return targets


def _handle_opened(event: CopyEvent, roster_state) -> None:
    pos = event.position
    ticker = pos.ticker
    if ticker is None:
        print(f"[{_now()}] {event.describe()} — SKIPPED (no broker route)")
        return
    if not pos.is_buy:
        print(f"[{_now()}] {event.describe()} — SKIPPED (shorts not handled yet)")
        return
    if pos.open_rate <= 0:
        print(f"[{_now()}] {event.describe()} — SKIPPED (no reference price to size from)")
        return

    targets = _target_accounts(event.username, ticker)
    if not targets:
        print(f"[{_now()}] {event.describe()} — SKIPPED (no linked account trades {ticker}'s asset class)")
        return

    broker_accounts = build_broker_accounts([a["id"] for a in targets])
    orders: list[AccountOrder] = []
    valid_accounts = []
    for broker_account in broker_accounts:
        clock = broker_account.get_market_clock()  # None for 24/7 venues (crypto) — nothing to check there
        if clock is not None and not clock.get("is_open", True):
            # Confirmed live 2026-09-17: submitting anyway doesn't error —
            # Alpaca ACCEPTS a market order while closed (success=True) but
            # doesn't fill it, so filled_qty comes back 0/None. Checking
            # first avoids ever recording a "bought" position (and sending a
            # misleading notification) for something that never actually
            # filled — the qty>0 guard below is belt-and-braces, not a
            # substitute for this.
            print(f"[{_now()}] {event.username}/{ticker} on {broker_account.nickname}: SKIPPED (market closed, next open {clock.get('next_open')})")
            continue
        try:
            qty = compute_qty_for_account(broker_account, pos.open_rate, COPY_SIZING_MODE, COPY_SIZING_VALUE, ticker)
        except Exception as e:  # noqa: BLE001 — one account's sizing failure can't skip the others
            print(f"[{_now()}] {event.username}/{ticker} on {broker_account.nickname}: sizing failed: {e}")
            continue
        orders.append(AccountOrder(account=broker_account, qty=qty))
        valid_accounts.append(broker_account)

    if not orders:
        return

    results = execute_order_across_accounts(orders, ticker, OrderSide.BUY)
    log_results(ticker, OrderSide.BUY, results)

    fills: list[CopyOrderFill] = []
    for broker_account, result in zip(valid_accounts, results):
        if not result.success:
            notifications.notify_order_rejected(ticker, "buy", broker_account.nickname, result.error or "unknown error")
            print(f"[{_now()}] {event.username}/{ticker} on {broker_account.nickname}: REJECTED — {result.error}")
            continue
        filled_qty = result.filled_qty or 0.0
        if filled_qty <= 0:
            # Accepted by the broker but not actually filled (see the market-
            # hours check above for the confirmed real case this catches) —
            # NOT recorded as an open position and NOT notified as "bought":
            # doing either would leave the roster thinking it holds something
            # it doesn't, with no way to ever close it out correctly.
            print(f"[{_now()}] {event.username}/{ticker} on {broker_account.nickname}: order accepted but not filled "
                  f"(qty=0) — not tracked, order id {result.broker_order_id}")
            continue
        fills.append(CopyOrderFill(
            account_id=broker_account.account_id, account_nickname=broker_account.nickname,
            qty=filled_qty, filled_avg_price=result.filled_avg_price,
        ))
        # broker_account.is_paper, NOT a hardcoded True — since AlpacaLive
        # became a valid target (2026-09-23), hardcoding True here would
        # have mislabelled a real live-money trade as paper on notification.
        notifications.notify_trade_open(
            ticker, f"Copy: {event.username}", filled_qty, broker_account.nickname, broker_account.is_paper,
        )
        print(f"[{_now()}] {event.username}/{ticker} on {broker_account.nickname}: BOUGHT {filled_qty}")

    if fills:
        roster_state.open_positions.append(OpenCopyPosition(
            username=event.username, etoro_position_id=pos.position_id, ticker=ticker,
            side="long", opened_at=event.detected_at, fills=fills,
        ))


def _handle_closed(event: CopyEvent, roster_state) -> None:
    pos = event.position
    open_pos = find_open_position(roster_state, pos.position_id)
    if open_pos is None:
        # Either a position that was already open before this executor started
        # tracking (opened before we ever polled it), or a short/unmapped one
        # we deliberately skipped on open — nothing of ours to close either way.
        print(f"[{_now()}] {event.describe()} — no Chopper position to close (never opened one)")
        return

    broker_accounts = build_broker_accounts([f.account_id for f in open_pos.fills])
    by_id = {b.account_id: b for b in broker_accounts}
    orders = [
        AccountOrder(account=by_id[f.account_id], qty=f.qty)
        for f in open_pos.fills if f.account_id in by_id
    ]
    if not orders:
        print(f"[{_now()}] {event.describe()} — WARNING: recorded fills but no matching account found")
        remove_open_position(roster_state, pos.position_id)
        return

    results = execute_order_across_accounts(orders, open_pos.ticker, OrderSide.SELL)
    log_results(open_pos.ticker, OrderSide.SELL, results)

    closed_account_ids: set[str] = set()
    for fill, result in zip(open_pos.fills, results):
        broker_account = by_id.get(fill.account_id)
        if broker_account is None:
            continue
        if not result.success:
            notifications.notify_order_rejected(open_pos.ticker, "sell", broker_account.nickname, result.error or "unknown error")
            print(f"[{_now()}] {event.username}/{open_pos.ticker} on {broker_account.nickname}: CLOSE REJECTED — {result.error}")
            continue
        filled_qty = result.filled_qty or 0.0
        if filled_qty <= 0:
            # Same "accepted but not actually filled" case as the open side
            # (e.g. market closed) — leave this fill in open_positions so
            # it's retried next cycle instead of being silently dropped
            # while Chopper still genuinely holds the shares.
            print(f"[{_now()}] {event.username}/{open_pos.ticker} on {broker_account.nickname}: close accepted but "
                  f"not filled (qty=0) — still open, will retry, order id {result.broker_order_id}")
            continue
        exit_price = result.filled_avg_price or 0.0
        entry_price = fill.filled_avg_price or 0.0
        strategy_tag = f"Copy: {event.username}"
        # broker_account.is_paper, NOT a hardcoded True — a live AlpacaLive
        # copy-trade close used to get recorded here as is_paper=True,
        # silently corrupting live_trades.db's live-vs-paper split (the
        # exact split this project has been comparing performance on).
        # Same fix as the open side above; see that note for why it started
        # mattering 2026-09-23.
        live_trades.record_realized_trade(
            account_id=broker_account.account_id, ticker=open_pos.ticker, strategy_name=strategy_tag,
            is_paper=broker_account.is_paper, entry_time=open_pos.opened_at, entry_price=entry_price,
            exit_time=datetime.now(timezone.utc).isoformat(), exit_price=exit_price, qty=filled_qty,
        )
        pnl = (exit_price - entry_price) * filled_qty if entry_price else 0.0
        notifications.notify_trade_close(
            open_pos.ticker, strategy_tag, filled_qty, pnl, broker_account.nickname, broker_account.is_paper,
        )
        print(f"[{_now()}] {event.username}/{open_pos.ticker} on {broker_account.nickname}: SOLD {filled_qty} (P&L ${pnl:+.2f})")
        closed_account_ids.add(fill.account_id)

    # Only drop the fills that actually closed — a partial close (some
    # accounts filled, others didn't) keeps the rest tracked for next cycle
    # rather than losing track of shares Chopper still genuinely holds.
    open_pos.fills = [f for f in open_pos.fills if f.account_id not in closed_account_ids]
    if not open_pos.fills:
        remove_open_position(roster_state, pos.position_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--username", required=True, action="append", help="eToro username to copy — repeat for multiple")
    parser.add_argument("--account-nickname", default=None, help="Which linked eToro account's keys to use")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)  # see copy_trader.py's identical note

    account_nickname, api_key, user_key = _find_etoro_credentials(args.account_nickname)
    roster_state = load_roster()
    ensure_entries(roster_state, args.username)
    save_roster(roster_state)

    usernames = active_usernames(roster_state)
    print(f"[{_now()}] EXECUTING copies for {len(usernames)} active investor(s) via '{account_nickname}': "
          f"{', '.join(usernames)}. Sizing: {COPY_SIZING_VALUE}% equity per position. Ctrl+C to stop.")

    while True:
        roster_state = load_roster()  # re-read: lets the roster be edited (paused/resumed) between cycles
        for username in active_usernames(roster_state):
            try:
                events = poll_investor(username, api_key, user_key)
            except CopyTradingError as e:
                print(f"[{_now()}] {username}: poll failed: {e}")
                continue
            except Exception as e:  # noqa: BLE001 — same "one investor's hiccup can't kill the loop" fix as copy_trader.py
                print(f"[{_now()}] {username}: poll failed (unexpected: {type(e).__name__}: {e})")
                continue
            if not events:
                # Silent-on-quiet-cycle was a real gap: an idle log looked
                # identical to a hung process, which defeats "watch this run
                # live and catch problems as they happen" — same visibility
                # copy_trader.py's observer already had, now matched here.
                print(f"[{_now()}] {username}: no change")
            for event in events:
                if event.action == "opened":
                    _handle_opened(event, roster_state)
                else:
                    _handle_closed(event, roster_state)
            save_roster(roster_state)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
