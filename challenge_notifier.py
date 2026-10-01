"""Trades a Kraken Funded challenge attempt BY NOTIFICATION, not execution -
tells the user exactly what to do, they place it themselves in the Kraken
app. Sibling to challenge_trader.py, built to replace it for this specific
product after confirming (2026-10-01, both live and from Kraken's own
support docs) that Kraken Funded/Prop has NO API access at all, for either
evaluation or funded accounts - every order has to be placed manually,
regardless of credentials. challenge_trader.py's whole execution path
(execute_order_across_accounts, position_attribution, live equity reads)
assumed broker access that turns out not to exist for this product; rather
than patch that script into something it was never designed to be, this is
a parallel, smaller tool built around the real constraint instead.

WHAT CHOPPER CAN AND CAN'T DO HERE. It CAN: fetch real BTC price data (the
regular Kraken API's public market-data endpoints work fine - they're not
part of what Kraken Funded restricts), compute the same strategy signal
auto_trader.py would, work out a suggested position size, and push a
notification the instant the signal changes. It CANNOT: see the real
challenge account's actual balance or holdings (no API for that - see
above), confirm a manually-placed order actually filled, or know if the
user skipped a notification or got a different fill price. Every number
this script tracks (estimated_equity, holding, entry_price) is therefore
an ESTIMATE built from "assume every notification was followed at
~the suggested price" - not ground truth. It drifts from the real account
the moment a trade is skipped, late, or filled at a different price, which
is why every outward-facing notification says "estimate" explicitly rather
than implying certainty the script has no way to actually have.

Price data comes from the regular KRKAPI account (--account-nickname) -
used ONLY for get_live_bars(), a public endpoint, never for balance/
position/order calls, since those would reflect the regular spot account,
not the challenge account.

Usage:
    python challenge_notifier.py --account-nickname KRKAPI --tier starter --interval 90
    (--tier only matters for starting a NEW attempt - ignored if one's already active)

To correct estimated_equity after checking the real balance in the Kraken
app (recommended periodically, since drift is expected - see above):
    python challenge_notifier.py --set-equity 1042.50
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import challenge_state, live_trades, notifications  # noqa: E402
from backtester.accounts import build_broker_accounts, list_accounts  # noqa: E402
from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Bar, Signal  # noqa: E402

TICKER = "X:BTCUSD"
STRATEGY_NAME = "VWAP Mean Reversion"  # matches challenge_trader.py's own choice - see that
# file's comment for why (min_bars comfortably inside Kraken's live-bars ceiling)
SIZING_PCT = 0.75  # the kraken_funded_challenge_sim.py result this whole ladder plan is built on
LOOKBACK_DAYS = 1
DEFAULT_INTERVAL_SECONDS = 90
ACCOUNT_ID = "kraken-funded-manual"  # deliberately NOT the real KRKAPI account id - this
# tracks a challenge balance that has nothing to do with that account's real one, and
# conflating the two account_ids would corrupt real P&L history with estimated numbers.

STATE_PATH = STATE_DIR / "challenge_notifier_state.json"


@dataclass
class NotifierState:
    holding: bool = False
    entry_price: float | None = None
    entry_time: str | None = None
    estimated_equity: float = 0.0


def _load_state(default_equity: float) -> NotifierState:
    raw = read_state_json(STATE_PATH, default=None)
    if raw is None:
        return NotifierState(estimated_equity=default_equity)
    return NotifierState(**raw)


def _save_state(state: NotifierState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(STATE_PATH, __import__("json").dumps(asdict(state), indent=2))


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _find_account(nickname: str):
    matches = [a for a in list_accounts() if a["nickname"] == nickname]
    if not matches:
        raise SystemExit(f"No linked account named {nickname!r}. Link it via the Accounts tab first.")
    return build_broker_accounts([matches[0]["id"]])[0]


def run_cycle(price_account, attempt: "challenge_state.ChallengeAttempt", state: NotifierState) -> bool:
    """Returns False if the attempt ended this cycle (estimated), True to
    keep going. Mutates and saves `state` as a side effect."""
    to_date = datetime.now(timezone.utc).date()
    from_date = to_date - timedelta(days=LOOKBACK_DAYS)
    bars = price_account.get_live_bars(
        ticker=TICKER, from_date=str(from_date), to_date=str(to_date), multiplier=1, timespan="minute",
    )
    if bars.empty or len(bars) < 6:
        print(f"[{_now()}] Not enough bars yet ({len(bars)}) - skipping this cycle.")
        return True

    strategy = build_strategy(STRATEGY_NAME)
    row = bars.iloc[-1]
    current = Bar(
        timestamp=bars.index[-1], open=row["open"], high=row["high"],
        low=row["low"], close=row["close"], volume=row["volume"],
    )
    signal = strategy.on_bar(bars, current)
    price = current.close

    progress = attempt.progress_fraction(state.estimated_equity)
    if signal is Signal.BUY and not state.holding:
        qty = (state.estimated_equity * SIZING_PCT) / price
        notifications.notify_manual_trade_suggestion(TICKER, "buy", qty, price, attempt.tier)
        state.holding = True
        state.entry_price = price
        state.entry_time = datetime.now(timezone.utc).isoformat()
        _save_state(state)
        print(f"[{_now()}] -> SUGGEST BUY ~{qty:.6f} {TICKER} @ ~${price:,.2f} "
              f"(est. equity ${state.estimated_equity:,.2f}, {progress * 100:.0f}% floor->target)")
        return True

    if signal is Signal.SELL and state.holding:
        entry = state.entry_price or price
        qty = (state.estimated_equity * SIZING_PCT) / entry
        pnl = (price - entry) * qty
        pnl_pct = (price - entry) / entry * 100 if entry else 0.0
        notifications.notify_manual_trade_suggestion(TICKER, "sell", qty, price, attempt.tier)
        notifications.notify_manual_trade_result(TICKER, pnl, pnl_pct)

        entry_time = state.entry_time or datetime.now(timezone.utc).isoformat()
        state.estimated_equity += pnl
        state.holding = False
        state.entry_price = None
        state.entry_time = None
        _save_state(state)

        live_trades.record_realized_trade(
            account_id=ACCOUNT_ID, ticker=TICKER, strategy_name=f"Challenge: {attempt.tier} (manual)",
            is_paper=True, entry_time=entry_time, entry_price=entry,
            exit_time=datetime.now(timezone.utc).isoformat(), exit_price=price, qty=qty,
        )
        print(f"[{_now()}] -> SUGGEST CLOSE ~{qty:.6f} {TICKER} @ ~${price:,.2f} "
              f"(est. P&L ${pnl:+.2f}, new est. equity ${state.estimated_equity:,.2f})")

        if state.estimated_equity >= attempt.target:
            notifications.notify_manual_challenge_ended("passed", attempt.tier, state.estimated_equity, attempt.target)
            print(f"[{_now()}] Estimated equity crossed TARGET - confirm in the Kraken app, "
                  f"then end the attempt (challenge_state.end_attempt).")
            return False
        if state.estimated_equity <= attempt.floor:
            notifications.notify_manual_challenge_ended("failed", attempt.tier, state.estimated_equity, attempt.floor)
            print(f"[{_now()}] Estimated equity crossed FLOOR - confirm in the Kraken app, "
                  f"then end the attempt (challenge_state.end_attempt).")
            return False
        return True

    print(f"[{_now()}] {TICKER} {STRATEGY_NAME} -> {signal.value.upper()} "
          f"(est. equity ${state.estimated_equity:,.2f}, {progress * 100:.0f}% floor->target, "
          f"{'holding' if state.holding else 'flat'})")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--account-nickname", help="Real Kraken account used ONLY for price data.")
    parser.add_argument("--tier", choices=list(challenge_state.TIERS), help="Start a new attempt at this tier if none is active.")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--set-equity", type=float, help="Correct the tracked estimate after checking the real Kraken app balance, then exit.")
    args = parser.parse_args()

    history = challenge_state.load()
    attempt = history.active

    if args.set_equity is not None:
        if attempt is None:
            raise SystemExit("No active attempt to correct.")
        state = _load_state(default_equity=attempt.starting_balance)
        old = state.estimated_equity
        state.estimated_equity = args.set_equity
        _save_state(state)
        print(f"Corrected estimated equity: ${old:,.2f} -> ${args.set_equity:,.2f}")
        return

    if attempt is None:
        if not args.tier:
            raise SystemExit("No active challenge attempt. Pass --tier starter|mid|anchor to start one.")
        attempt = challenge_state.start_attempt(args.tier, ACCOUNT_ID, STRATEGY_NAME, SIZING_PCT)
        print(f"[{_now()}] Started attempt #{attempt.attempt_number} ({attempt.tier}): "
              f"${attempt.starting_balance:,.0f} start, target ${attempt.target:,.2f}, floor ${attempt.floor:,.2f}.")

    if not args.account_nickname:
        raise SystemExit("--account-nickname is required (a real Kraken account, used only for price data).")
    price_account = _find_account(args.account_nickname)
    state = _load_state(default_equity=attempt.starting_balance)

    print(f"[{_now()}] Notifying for attempt #{attempt.attempt_number} ({attempt.tier}): "
          f"target ${attempt.target:,.2f}, floor ${attempt.floor:,.2f}, sizing {SIZING_PCT*100:.0f}% per entry. "
          f"Est. equity ${state.estimated_equity:,.2f}. Every trade is a NOTIFICATION, not a real order - "
          f"place it yourself. Ctrl+C to stop.")

    while True:
        keep_going = run_cycle(price_account, attempt, state)
        if not keep_going:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
