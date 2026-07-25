"""Standalone auto-trading process.

Runs independently of the dashboard — start it from a terminal
(`python auto_trader.py`) or let the dashboard launch it as a detached
subprocess. Either way, it coordinates through auto_trader_state.py's
control/status files, so the dashboard can arm/disarm/kill it and see its
status regardless of who started it.

Safety model:
- Re-checks the kill switch and enabled flag before every ticker, not just
  once per cycle or once at startup.
- A missing/corrupt control file is treated as killed, not "keep trading
  with stale settings" (see auto_trader_state.load_control).
- Enforces max_trades_per_day; stops issuing new orders once hit.
- Only targets accounts explicitly listed in the control file, and further
  filters out live accounts unless allow_live is set.
- Won't BUY a ticker an account already holds, won't SELL one it doesn't
  hold (mirrors the backtest engine's flat-position-only entries) — checked
  fresh via get_positions() every time, not cached.
- Sizing failures, data failures, and position-check failures for one
  ticker/account never abort the whole cycle — they're logged to status
  and the loop moves on.

Adaptive roster mode (control.use_roster=True): instead of one fixed
strategy_name/tickers pair, trades whatever backtester.roster.py's active
(and paused, exit-only) entries say. Every BUY/SELL, in both manual and
roster mode, is attributed to its strategy via position_attribution.py and,
once closed, recorded as a realized trade in live_trades.py — this is the
real-trade-outcome history the roster's demotion rules read from. Roster-mode
demotion checks (pausing something that's actually losing money right now)
run every cycle; promotion only happens via a deliberate "re-evaluate roster"
action from the dashboard (scanning is too expensive to trigger every poll).

Account-level max-drawdown circuit breaker (control.max_drawdown_enabled):
independent of manual/roster mode — every cycle, each target account's
current equity is checked against its own peak equity via account_risk.py.
Once an account's drawdown from its peak exceeds max_drawdown_pct, that
account is hard-blocked from new entries (existing positions can still be
closed) and STAYS blocked, even if equity recovers, until a human manually
re-arms it from the dashboard. This is deliberately not self-healing.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from backtester import account_risk, accounts as accounts_module
from backtester import live_trades, notifications, position_attribution, roster, volatility
from backtester.auto_trader_state import AutoTraderStatus, load_control, load_status, save_status
from backtester.brokers.base import BrokerAccount, OrderSide
from backtester.data import PolygonClient, PolygonError
from backtester.execution import AccountOrder, SizingMode, compute_qty_for_account, execute_order_across_accounts
from backtester.execution_log import log_results
from backtester.conviction import compute_conviction
from backtester.strategies import STRATEGY_REGISTRY, build_strategy
from backtester.strategy import Bar

LOOKBACK_DAYS = 90  # enough history for any strategy's default window, even on daily bars
VOL_REGIME_LOOKBACK_DAYS = 1100  # daily-bar history fetched for the GARCH regime

# In-process cache: {ticker: (date_computed, RegimeInfo | None)}. The regime is a
# daily-granularity concept, so it's only worth recomputing once per calendar day
# per ticker, not on every poll cycle (which can be as frequent as every 5s).
_regime_cache: dict[str, tuple[str, "volatility.RegimeInfo | None"]] = {}


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _get_regime(daily_client: PolygonClient, ticker: str, target_vol_ann: float) -> "volatility.RegimeInfo | None":
    today = _today_str()
    cached = _regime_cache.get(ticker)
    if cached is not None and cached[0] == today:
        return cached[1]

    try:
        daily_from = (datetime.now(timezone.utc).date() - timedelta(days=VOL_REGIME_LOOKBACK_DAYS)).isoformat()
        daily_bars = daily_client.get_aggregates(
            ticker=ticker, from_date=daily_from, to_date=_today_str(), multiplier=1, timespan="day",
        )
        table = volatility.compute_regime_table(daily_bars, target_vol_ann=target_vol_ann)
        info = volatility.latest_regime_info(table)
    except (PolygonError, volatility.InsufficientHistoryError):
        info = None  # not enough daily history (or a fetch failure) -> trade unfiltered

    _regime_cache[ticker] = (today, info)
    return info


def _resolve_targets(control) -> list[tuple[str, str, dict, bool]]:
    """Returns (ticker, strategy_name, params, no_new_entries) tuples to trade
    this cycle. Manual mode: one entry per control.tickers, never blocked from
    new entries. Roster mode: runs the cheap demotion-only pass (never
    promotes — that's a deliberate, separate dashboard action) and trades
    every active AND paused entry, so a paused entry can still exit an
    existing position, just never open a new one.
    """
    if not control.use_roster:
        return [(ticker, control.strategy_name, {}, False) for ticker in control.tickers]

    state = roster.load_roster()
    state = roster.apply_demotion_checks(state, live_trades.recent_performance, state.config)
    roster.save_roster(state)

    return [
        (entry.ticker, entry.strategy_name, entry.params, entry.status == "paused")
        for entry in state.entries
        if entry.status in ("active", "paused")
    ]


def _trade_target(
    ticker: str,
    strategy_name: str,
    params: dict,
    no_new_entries: bool,
    broker_accounts: list[BrokerAccount],
    client: PolygonClient,
    daily_client: PolygonClient,
    control,
    status: AutoTraderStatus,
    blocked_account_ids: set[str],
    market_closed_account_ids: set[str],
) -> None:
    """Trade one (ticker, strategy) pair for this cycle: fetch bars, get a
    signal, apply the GARCH vol-target filter/sizing, and submit orders
    across every target account, attributing and recording each realized
    round trip. Mutates `status` in place.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        status.last_error = f"unknown strategy '{strategy_name}' — skipping {ticker}"
        return

    try:
        bars = client.get_aggregates(
            ticker=ticker,
            from_date=(datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)).isoformat(),
            to_date=_today_str(),
            multiplier=control.multiplier,
            timespan=control.timespan,
        )
    except Exception as e:  # noqa: BLE001
        status.last_error = f"{ticker}: data fetch failed: {e}"
        return

    if bars.empty or len(bars) < 2:
        return

    strategy = build_strategy(strategy_name, params=params)
    row = bars.iloc[-1]
    current = Bar(
        timestamp=bars.index[-1],
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
    )
    signal = strategy.on_bar(bars, current)

    if signal.value == "hold":
        return

    size_multiplier = 1.0
    if control.vol_target_enabled:
        regime_info = _get_regime(daily_client, ticker, control.vol_target_ann)
        if regime_info is not None:
            if signal.value == "buy" and regime_info.regime == "storm":
                status.last_signal = (
                    f"{ticker}/{strategy_name}: BUY blocked — GARCH storm regime "
                    f"(vol_pctile={regime_info.vol_pctile:.0f})"
                )
                save_status(status)
                return
            size_multiplier = regime_info.size_multiplier

    status.last_signal = f"{ticker}/{strategy_name}: {signal.value.upper()} @ {current.close}"
    save_status(status)

    order_side = OrderSide.BUY if signal.value == "buy" else OrderSide.SELL
    sizing_mode = SizingMode(control.sizing_mode)
    # Score the entry signal's conviction once (#25). Only meaningful for a BUY
    # (a new entry); carried through attribution to live_trades on the eventual
    # close. Logged only — it does NOT influence sizing/gating anywhere.
    entry_conviction = compute_conviction(strategy, bars, current) if order_side is OrderSide.BUY else None

    account_orders: list[AccountOrder] = []
    # parallel to account_orders — pre-submit position snapshot per account, for
    # attribution (BUY) and realized-P&L computation (SELL) after the fill.
    order_contexts: list[dict] = []

    for broker_account in broker_accounts:
        if broker_account.account_id in market_closed_account_ids:
            continue  # market shut for this account — never trade on stale bars

        try:
            positions = broker_account.get_positions()
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: position check failed: {e}"
            continue

        held = {p.ticker for p in positions if p.qty > 0}
        position_attribution.reconcile(broker_account.account_id, held)
        existing_position = next((p for p in positions if p.ticker == ticker and p.qty > 0), None)
        has_position = existing_position is not None

        # A submitted-but-unfilled order isn't a position yet, so the has_position
        # check alone can't stop us re-ordering the same ticker before the first
        # fill. Skip any ticker that already has an open order on this account —
        # covers slow fills during hours, not just the overnight case above.
        try:
            if ticker in broker_account.get_open_order_tickers():
                continue
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: open-order check failed: {e}"
            continue

        if order_side is OrderSide.BUY:
            if has_position or no_new_entries or broker_account.account_id in blocked_account_ids:
                continue
        elif order_side is OrderSide.SELL and not has_position:
            continue

        try:
            effective_sizing_value = control.sizing_value * size_multiplier
            qty = compute_qty_for_account(broker_account, current.close, sizing_mode, effective_sizing_value)
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: sizing failed: {e}"
            continue

        account_orders.append(AccountOrder(account=broker_account, qty=qty))
        order_contexts.append({"account": broker_account, "existing_position": existing_position})

    if not account_orders:
        return

    results = execute_order_across_accounts(account_orders, ticker, order_side)
    log_results(ticker, order_side, results)
    status.trades_today += 1
    save_status(status)

    for ctx, result in zip(order_contexts, results):
        if not result.success:
            continue
        broker_account = ctx["account"]
        if order_side is OrderSide.BUY:
            position_attribution.record_open(
                broker_account.account_id, ticker, strategy_name, conviction=entry_conviction
            )
            notifications.notify_trade_open(
                ticker, strategy_name, result.filled_qty or 0.0,
                broker_account.nickname, broker_account.is_paper,
            )
        else:
            attribution = position_attribution.pop_open(broker_account.account_id, ticker)
            if attribution is None:
                continue  # position wasn't opened through this flow — no track record to close out
            existing_position = ctx["existing_position"]
            exit_price = result.filled_avg_price or current.close
            qty = result.filled_qty or existing_position.qty
            live_trades.record_realized_trade(
                account_id=broker_account.account_id,
                ticker=ticker,
                strategy_name=attribution["strategy_name"],
                is_paper=broker_account.is_paper,
                entry_time=attribution["opened_at"],
                entry_price=existing_position.avg_entry_price,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
                qty=qty,
                conviction=attribution.get("conviction"),
            )
            pnl = (exit_price - existing_position.avg_entry_price) * qty
            notifications.notify_trade_close(
                ticker, attribution["strategy_name"], qty, pnl,
                broker_account.nickname, broker_account.is_paper,
            )


def run_cycle(status: AutoTraderStatus) -> AutoTraderStatus:
    control = load_control()

    status.running = True
    status.pid = os.getpid()
    status.last_heartbeat = datetime.now(timezone.utc).isoformat()

    today = _today_str()
    if status.trades_date != today:
        status.trades_date = today
        status.trades_today = 0

    if control.killed:
        status.last_signal = "killed — not trading"
        save_status(status)
        return status

    if not control.enabled:
        status.last_signal = "disabled — not trading"
        save_status(status)
        return status

    if status.trades_today >= control.max_trades_per_day:
        status.last_signal = f"max trades/day ({control.max_trades_per_day}) reached — not trading"
        save_status(status)
        return status

    if not control.account_ids or (
        not control.use_roster and (not control.tickers or not control.strategy_name)
    ):
        status.last_error = "control missing tickers/strategy/accounts — not trading"
        save_status(status)
        return status

    try:
        client = PolygonClient(use_cache=False)  # must always see fresh bars, not the day's first poll
        daily_client = PolygonClient(use_cache=True)  # daily-bar GARCH regime data is fine to cache
    except Exception as e:  # noqa: BLE001
        status.last_error = f"Polygon client init failed: {e}"
        save_status(status)
        return status

    try:
        linked = accounts_module.list_accounts()
        target_meta = [a for a in linked if a["id"] in control.account_ids]
        if not control.allow_live:
            target_meta = [a for a in target_meta if a["is_paper"]]
        if not target_meta:
            status.last_error = "no eligible target accounts (check allow_live / linked accounts)"
            save_status(status)
            return status
        broker_accounts = accounts_module.build_broker_accounts([a["id"] for a in target_meta])
    except Exception as e:  # noqa: BLE001
        status.last_error = f"failed to build broker accounts: {e}"
        save_status(status)
        return status

    status.last_error = None

    # Market-hours guard: never trade an account whose market is closed. Overnight,
    # the data feed keeps returning the previous session's final bar (stale), so a
    # mean-reversion strategy re-fires the same signal every cycle and — because a
    # queued, unfilled order isn't yet a position — stacks duplicate orders until the
    # daily cap. Skip closed accounts entirely; a broker with no clock (24/7 crypto)
    # returns None and is always considered open.
    market_closed_account_ids: set[str] = set()
    next_open = None
    for broker_account in broker_accounts:
        try:
            clock = broker_account.get_market_clock()
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: market clock check failed: {e}"
            continue
        if clock is not None and not clock["is_open"]:
            market_closed_account_ids.add(broker_account.account_id)
            next_open = clock.get("next_open")

    if len(market_closed_account_ids) == len(broker_accounts):
        when = f" (next open {next_open:%A %H:%M} ET)" if next_open is not None else ""
        status.last_signal = f"market closed — not trading{when}"
        save_status(status)
        return status

    blocked_account_ids: set[str] = set()
    if control.max_drawdown_enabled:
        for broker_account in broker_accounts:
            try:
                equity = broker_account.get_account_snapshot().equity
                blocked, reason = account_risk.check_and_update(
                    broker_account.account_id, equity, control.max_drawdown_pct
                )
            except Exception as e:  # noqa: BLE001
                status.last_error = f"{broker_account.nickname}: account risk check failed: {e}"
                continue
            if blocked:
                blocked_account_ids.add(broker_account.account_id)
                status.last_error = f"{broker_account.nickname}: account risk limit breached — {reason}"

    targets = _resolve_targets(control)
    if not targets:
        status.last_signal = "roster empty — nothing to trade" if control.use_roster else status.last_signal
        save_status(status)
        return status

    for ticker, strategy_name, params, no_new_entries in targets:
        control = load_control()  # re-check every target, not just once per cycle
        if control.killed or not control.enabled:
            status.last_signal = "stopped mid-cycle (killed or disabled)"
            save_status(status)
            return status
        if status.trades_today >= control.max_trades_per_day:
            break

        _trade_target(
            ticker, strategy_name, params, no_new_entries,
            broker_accounts, client, daily_client, control, status,
            blocked_account_ids, market_closed_account_ids,
        )

    save_status(status)
    return status


def _another_instance_alive() -> bool:
    """True if another auto-trader already looks alive — a fresh heartbeat from a
    recent cycle. Guards against running two instances at once (which would
    double every order), now that the trader can be auto-started on login as
    well as launched from the dashboard. A stale heartbeat (dead/slept process)
    is treated as free to take over."""
    status = load_status()
    hb = status.last_heartbeat
    if not hb or not status.running:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
    except ValueError:
        return False
    control = load_control()
    # a live loop beats at least once per poll interval; allow ~3 cycles of slack
    return age < max(300, 3 * control.poll_interval_seconds) and status.pid != os.getpid()


def main() -> None:
    # Load .env by ABSOLUTE path (next to this file), not via cwd — so the bot
    # works identically whether launched from the dashboard, a terminal, or the
    # login auto-start task (which runs from an arbitrary working directory).
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if _another_instance_alive():
        print("Another auto-trader instance appears to be running (fresh heartbeat) — exiting to avoid double-trading.")
        return
    status = load_status()
    print(f"Auto-trader starting (pid={os.getpid()}).")
    try:
        while True:
            control = load_control()
            status = run_cycle(status)
            if control.killed:
                print("Kill switch engaged — stopping.")
                status.running = False
                save_status(status)
                break
            time.sleep(max(control.poll_interval_seconds, 5))
    except KeyboardInterrupt:
        status.running = False
        save_status(status)
        print("Stopped by user.")


if __name__ == "__main__":
    main()
