"""RETIRED 2026-10-01 - DOES NOT WORK, DO NOT RUN. Kept only for its
execution-path reasoning (see below), which is still accurate for every
OTHER account in this project - just not for this one.

Confirmed (both live, by querying the regular Kraken API with real linked
credentials - Balance/TradeBalance show only the normal spot account,
nothing resembling a challenge balance - and from Kraken's own support
docs: "Kraken Prop does not currently offer API access... all orders on a
Prop account are placed manually"): Kraken Funded/Prop has NO API access
at all, for either evaluation or funded accounts, regardless of
credentials. This script's entire premise - executing real orders against
a Kraken Funded account via the broker API - is therefore impossible, not
just unconfirmed as originally written below. See challenge_state.py's own
updated docstring for the full finding.

REPLACEMENT: challenge_notifier.py. Same signal computation, same
challenge_state.py tracking, but it NOTIFIES the suggested trade instead
of executing it - the user places each order by hand in the Kraken app.
Not a downgrade born of giving up on automation; it's the actual shape
this product allows.

Original module docstring, kept for the reasoning trail (still correct
about every account OTHER than Kraken Funded):

WHY SEPARATE, NOT A SPECIAL CASE IN auto_trader.py: every other account
Chopper trades has an open-ended mandate (be profitable long-term) with
only DOWNSIDE circuit breakers (max_drawdown, daily giveback). A challenge
attempt races between a fixed +12% TARGET and a fixed -3% FLOOR, both
measured from a single starting point, and ends the INSTANT either is hit -
there is no "keep trading, you're doing fine" state once you've crossed
either line, and nothing about that resembles roster-mode's continuous
demotion/promotion logic. Bolting a hard upside stop onto auto_trader.py's
loop risked breaking the invariant every OTHER account there relies on
(never stop because you're winning). Separate process, separate state file
(challenge_state.py), same proven execution path underneath
(execute_order_across_accounts, compute_qty_for_account,
position_attribution) - reused, not reimplemented, same as
copy_trade_executor.py's own reasoning for reusing auto_trader.py's path
rather than rebuilding it.

NO ADVANCED ORDER TYPES, NO LEVERAGE (Kraken Funded's own stated rules) -
this deliberately never passes take_profit_price/stop_loss_price to
AccountOrder. Plain market orders only - moot now, but was a real
constraint while this script's premise still looked viable.

Usage (RETIRED - do not run):
    python challenge_trader.py --account-nickname <linked-account> --interval 90
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import challenge_state, live_trades, notifications, position_attribution  # noqa: E402
from backtester.accounts import build_broker_accounts, list_accounts  # noqa: E402
from backtester.brokers.base import OrderSide  # noqa: E402
from backtester.execution import (  # noqa: E402
    AccountOrder,
    SizingMode,
    compute_qty_for_account,
    execute_order_across_accounts,
)
from backtester.execution_log import log_results  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Bar, Signal  # noqa: E402

TICKER = "X:BTCUSD"
STRATEGY_NAME = "VWAP Mean Reversion"  # min_bars=5 - comfortably inside Kraken's
# own get_live_bars() ceiling of ~12h/721 one-minute candles (see kraken.py's
# own docstring for that confirmed, non-Coinbase-backed limitation). A
# strategy needing a genuinely multi-day lookback would need revisiting this.
LOOKBACK_DAYS = 1  # get_live_bars ignores `from_date` beyond ~12h anyway - see above
DEFAULT_INTERVAL_SECONDS = 90


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _find_account(nickname: str):
    matches = [a for a in list_accounts() if a["nickname"] == nickname]
    if not matches:
        raise SystemExit(f"No linked account named {nickname!r}. Link it via the Accounts tab first.")
    return build_broker_accounts([matches[0]["id"]])[0]


def run_cycle(broker_account, attempt: "challenge_state.ChallengeAttempt") -> bool:
    """Returns False if the attempt ended this cycle (caller should stop
    polling), True to keep going."""
    snapshot = broker_account.get_account_snapshot()
    equity = snapshot.equity

    if equity >= attempt.target:
        challenge_state.end_attempt("passed", equity)
        notifications.notify(
            f"Kraken Funded challenge PASSED ({attempt.tier})",
            f"Attempt #{attempt.attempt_number}: reached ${equity:,.2f}, target was ${attempt.target:,.2f}. "
            "No further trading on this attempt - see Kraken app for next steps.",
            priority="high",
        )
        print(f"[{_now()}] PASSED — equity ${equity:,.2f} >= target ${attempt.target:,.2f}. Stopping.")
        return False

    if equity <= attempt.floor:
        challenge_state.end_attempt("failed", equity)
        notifications.notify(
            f"Kraken Funded challenge FAILED ({attempt.tier})",
            f"Attempt #{attempt.attempt_number}: dropped to ${equity:,.2f}, floor was ${attempt.floor:,.2f}. "
            "Fee is spent - start a new attempt when ready.",
            priority="high",
        )
        print(f"[{_now()}] FAILED — equity ${equity:,.2f} <= floor ${attempt.floor:,.2f}. Stopping.")
        return False

    # broker_account.get_live_bars(), NOT a separate Polygon fetch - matches
    # auto_trader.py's own established live-data-first path (Polygon pulled
    # out of the live signal path entirely, 2026-08-20 - see that file's own
    # comment for the staleness bug this avoids repeating).
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - __import__("datetime").timedelta(days=LOOKBACK_DAYS)
    bars = broker_account.get_live_bars(
        ticker=TICKER, from_date=str(from_date), to_date=str(to_date), multiplier=1, timespan="minute",
    )
    if bars.empty or len(bars) < 6:  # strategy's own min_bars=5, plus the current bar
        print(f"[{_now()}] Not enough bars yet ({len(bars)}) - skipping this cycle.")
        return True

    strategy = build_strategy(STRATEGY_NAME)
    row = bars.iloc[-1]
    current = Bar(
        timestamp=bars.index[-1], open=row["open"], high=row["high"],
        low=row["low"], close=row["close"], volume=row["volume"],
    )
    # The FULL bars frame (current row included), not history-excluding-
    # current - matches auto_trader.py's exact call, confirmed by reading
    # it rather than assumed (an earlier draft of this file got this wrong).
    signal = strategy.on_bar(bars, current)

    existing_positions = {p.ticker: p for p in broker_account.get_positions()}
    holding = TICKER in existing_positions and existing_positions[TICKER].qty > 0

    order_side = None
    if signal is Signal.BUY and not holding:
        order_side = OrderSide.BUY
    elif signal is Signal.SELL and holding:
        order_side = OrderSide.SELL

    if order_side is None:
        print(f"[{_now()}] {TICKER} {STRATEGY_NAME} -> {signal.value.upper()} (equity ${equity:,.2f}, "
              f"{(attempt.progress_fraction(equity) * 100):.0f}% of the way from floor to target)")
        return True

    try:
        if order_side is OrderSide.BUY:
            qty = compute_qty_for_account(
                broker_account, current.close, SizingMode.PCT_EQUITY, attempt.sizing_pct * 100, TICKER,
            )
        else:
            qty = existing_positions[TICKER].qty
    except Exception as e:  # noqa: BLE001
        print(f"[{_now()}] sizing failed: {e}")
        return True

    # Deliberately NO take_profit_price/stop_loss_price - Kraken Funded's
    # own rules: no advanced order types.
    order = AccountOrder(account=broker_account, qty=qty)
    results = execute_order_across_accounts([order], TICKER, order_side)
    log_results(TICKER, order_side, results)
    result = results[0]

    if not result.success:
        notifications.notify_order_rejected(TICKER, order_side.value, broker_account.nickname, result.error or "unknown error")
        print(f"[{_now()}] {order_side.value.upper()} REJECTED — {result.error}")
        return True

    filled_qty = result.filled_qty or 0.0
    if filled_qty <= 0:
        print(f"[{_now()}] order accepted but not filled (qty=0) — order id {result.broker_order_id}")
        return True

    if order_side is OrderSide.BUY:
        position_attribution.record_open(
            broker_account.account_id, TICKER, STRATEGY_NAME,
            sizing_mode=SizingMode.PCT_EQUITY.value, sizing_value=attempt.sizing_pct * 100,
            dollars_committed=filled_qty * current.close, entry_price=result.filled_avg_price or current.close,
        )
        notifications.notify_trade_open(TICKER, f"Challenge: {attempt.tier}", filled_qty, broker_account.nickname, broker_account.is_paper)
        print(f"[{_now()}] BOUGHT {filled_qty} {TICKER}")
    else:
        attribution = position_attribution.pop_open(broker_account.account_id, TICKER) or {
            "strategy_name": STRATEGY_NAME, "opened_at": "", "conviction": None,
        }
        entry_price = position_attribution.resolve_entry_price(attribution, existing_positions[TICKER].avg_entry_price)
        exit_price = result.filled_avg_price or current.close
        live_trades.record_realized_trade(
            account_id=broker_account.account_id, ticker=TICKER, strategy_name=f"Challenge: {attempt.tier}",
            is_paper=broker_account.is_paper, entry_time=attribution["opened_at"], entry_price=entry_price,
            exit_time=datetime.now(timezone.utc).isoformat(), exit_price=exit_price, qty=filled_qty,
        )
        pnl = (exit_price - entry_price) * filled_qty
        notifications.notify_trade_close(TICKER, f"Challenge: {attempt.tier}", filled_qty, pnl, broker_account.nickname, broker_account.is_paper)
        print(f"[{_now()}] SOLD {filled_qty} {TICKER} (P&L ${pnl:+.2f})")

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-nickname", required=True)
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    args = parser.parse_args()

    broker_account = _find_account(args.account_nickname)

    history = challenge_state.load()
    attempt = history.active
    if attempt is None:
        raise SystemExit(
            "No active challenge attempt. Start one first via "
            "challenge_state.start_attempt(tier, account_id, strategy_name, sizing_pct)."
        )

    print(f"[{_now()}] Trading attempt #{attempt.attempt_number} ({attempt.tier}): "
          f"${attempt.starting_balance:,.0f} start, target ${attempt.target:,.2f}, floor ${attempt.floor:,.2f}, "
          f"sizing {attempt.sizing_pct*100:.0f}% per entry. Ctrl+C to stop.")

    while True:
        keep_going = run_cycle(broker_account, attempt)
        if not keep_going:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
