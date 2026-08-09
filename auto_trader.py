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

Daily P&L giveback guard (control.giveback_enabled): a lighter, DAILY
counterpart to the breaker above — see daily_pnl_guard.py. Once an
account's profit for today has retraced more than giveback_pct off today's
own intraday peak, new entries block for the rest of the day (existing
positions can still close); unlike the breaker above, this resets itself
automatically at the start of the next day, no manual re-arm needed. Both
share the same blocked_account_ids set and the same new-entries-block /
sells-still-allowed behavior in _trade_target — the giveback check is
skipped for an account the breaker already blocked, since it'd be
redundant.

Broker instances are now cached across poll cycles (_get_broker_accounts,
_broker_cache), not rebuilt fresh every cycle like before. Harmless for
brokers with no per-object session (Alpaca/Coinbase/Kraken/Tastytrade's
simple key auth, IBKR's own deliberate connect-per-operation design), but
IGBroker holds ONE authenticated session per instance (ig.py — IG tolerates
only one fresh login per short window) — without this, a fresh instance
every poll cycle would mean a fresh IG login every poll cycle.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from backtester import account_risk, accounts as accounts_module, daily_pnl_guard
from backtester import current_signals, events, heartbeat, live_trades, notifications, position_attribution, roster, volatility
from backtester.auto_trader_state import (
    AutoTraderStatus, load_control, load_control_checked, load_status, save_status,
)
from backtester.brokers.base import BrokerAccount, OrderSide
from backtester.data import PolygonClient, PolygonError
from backtester.execution import (
    AccountOrder,
    SizingMode,
    compute_qty_for_account,
    execute_order_across_accounts,
    sliding_pct_equity,
)
from backtester.execution_log import log_results
from backtester.conviction import compute_conviction, compute_levels
from backtester.oanda_data import OandaDataClient, OandaError
from backtester.strategies import STRATEGY_REGISTRY, build_strategy
from backtester.strategy import Bar

LOOKBACK_DAYS = 90  # enough history for any strategy's default window, even on daily bars
RECENT_CLOSES_COUNT = 30  # trailing closes published for the mobile app's sparkline (see current_signals.py)
VOL_REGIME_LOOKBACK_DAYS = 1100  # daily-bar history fetched for the GARCH regime

# In-process cache: {ticker: (date_computed, RegimeInfo | None)}. The regime is a
# daily-granularity concept, so it's only worth recomputing once per calendar day
# per ticker, not on every poll cycle (which can be as frequent as every 5s).
_regime_cache: dict[str, tuple[str, "volatility.RegimeInfo | None"]] = {}

# In-process cache: {account_id: BrokerAccount}. run_cycle() used to call
# accounts_module.build_broker_accounts() fresh every single cycle — harmless
# for brokers that don't hold a session on the object (Alpaca/Coinbase/Kraken/
# Tastytrade's simple key auth, and IBKR's own deliberate connect-per-operation
# design), but IGBroker now caches ONE authenticated session per INSTANCE (see
# ig.py — IG tolerates only one fresh login per short window), so a fresh
# instance every poll cycle would mean a fresh login every poll cycle, exactly
# the failure mode that session-reuse fix exists to prevent. Reusing broker
# objects across cycles here is what actually makes that fix effective for
# continuous unattended running, not just for a single script/instance.
_broker_cache: dict[str, BrokerAccount] = {}


def _get_broker_accounts(account_ids: list[str]) -> list[BrokerAccount]:
    """Cached across poll cycles — see _broker_cache above. Only (re)builds an
    instance for an account_id not already cached; drops (and cleanly closes,
    for brokers like IG that hold a session) any cached instance for an
    account_id no longer requested, e.g. after a control.json edit."""
    stale_ids = set(_broker_cache) - set(account_ids)
    for aid in stale_ids:
        broker = _broker_cache.pop(aid)
        close = getattr(broker, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass  # best-effort cleanup only, never let this block a cycle

    missing_ids = [aid for aid in account_ids if aid not in _broker_cache]
    if missing_ids:
        for broker in accounts_module.build_broker_accounts(missing_ids):
            _broker_cache[broker.account_id] = broker

    return [_broker_cache[aid] for aid in account_ids if aid in _broker_cache]


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
    this cycle. Manual mode (primary): one entry per control.tickers, never
    blocked from new entries, params from control.manual_strategy_params
    (merged over STRATEGY_REGISTRY defaults inside build_strategy — empty
    dict here means "use the current default", same as always). Roster mode
    (primary): runs the cheap demotion-only pass (never promotes — that's a
    deliberate, separate dashboard action) and trades every active AND
    paused entry, so a paused entry can still exit an existing position,
    just never open a new one.

    Then, regardless of which primary branch ran, appends one target per
    ticker for every group in control.extra_targets — additional CONCURRENT
    manual-style targets (e.g. a forex account trading its own separately-
    tuned params alongside an equities roster primary). These are never
    blocked from new entries either. Routing to the right accounts relies
    entirely on the caller's existing per-(ticker,account) asset-class
    guard, not on anything here — this function just returns tickers.
    """
    if not control.use_roster:
        primary = [
            (ticker, control.strategy_name, control.manual_strategy_params.get(control.strategy_name, {}), False)
            for ticker in control.tickers
        ]
    else:
        state = roster.load_roster()
        state = roster.apply_demotion_checks(state, live_trades.recent_performance, state.config)
        roster.save_roster(state)

        primary = [
            (entry.ticker, entry.strategy_name, entry.params, entry.status == "paused")
            for entry in state.entries
            if entry.status in ("active", "paused")
        ]

    extra = [
        (ticker, group.get("strategy_name", ""), group.get("strategy_params", {}), False)
        for group in control.extra_targets
        for ticker in group.get("tickers", [])
    ]
    return primary + extra


_oanda_client: OandaDataClient | None | bool = False  # False = not checked yet, None = checked, unavailable


def _get_oanda_client() -> OandaDataClient | None:
    """Lazily built, cached for the process lifetime — OANDA is a plain data
    client (no session/auth handshake to keep warm like IG's), so the only
    reason to cache is avoiding re-reading the env var and re-raising
    OandaError every single cycle once we already know it's unconfigured.
    Returns None (falls through to Polygon) until OANDA_API_KEY is set —
    expected to be unset until the user generates the key, see
    CLAUDE_NOTES.txt."""
    global _oanda_client
    if _oanda_client is False:
        try:
            _oanda_client = OandaDataClient()
        except OandaError:
            _oanda_client = None
    return _oanda_client


def _pick_live_data_source(
    ticker: str, broker_accounts: list[BrokerAccount], account_asset_classes: dict[str, frozenset[str]],
) -> BrokerAccount | OandaDataClient | None:
    """Find a live-data source for this ticker, preferring (in order): OANDA
    for forex (a data-only client, never a target account — see
    CLAUDE_NOTES.txt for why: IG's own historical-price endpoint has a real
    weekly allowance too scarce for a 120s polling loop, one test pull
    exhausted it), then the first target account that both (a) can trade
    this ticker's asset class and (b) exposes get_live_bars — currently only
    AlpacaBroker (equities). Not account_asset_class-eligibility related
    (drawdown-blocked / market-closed accounts are still fine to READ data
    from, just not to trade on) — deliberately ignores blocked_account_ids/
    market_closed_account_ids, unlike the order-routing loop below.

    Returns None (falls through to Polygon) when nothing above is available
    — e.g. OANDA_API_KEY isn't set yet, or the ticker is neither forex nor
    covered by a live-data-capable broker account.
    """
    ticker_asset_class = accounts_module.infer_asset_class(ticker)

    if ticker_asset_class == "forex":
        oanda = _get_oanda_client()
        if oanda is not None:
            return oanda

    for broker_account in broker_accounts:
        if ticker_asset_class not in account_asset_classes.get(broker_account.account_id, frozenset()):
            continue
        if hasattr(broker_account, "get_live_bars"):
            return broker_account
    return None


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
    account_asset_classes: dict[str, frozenset[str]],
) -> None:
    """Trade one (ticker, strategy) pair for this cycle: fetch bars, get a
    signal, apply the GARCH vol-target filter/sizing, and submit orders
    across every target account, attributing and recording each realized
    round trip. Mutates `status` in place.
    """
    if strategy_name not in STRATEGY_REGISTRY:
        status.last_error = f"unknown strategy '{strategy_name}' — skipping {ticker}"
        return

    from_date = (datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    to_date = _today_str()

    # Prefer live data (Alpaca for equities, OANDA for forex once configured
    # — see _pick_live_data_source) over Polygon: this account's Polygon plan
    # has NO same-day intraday data at all (confirmed 2026-08-05, see
    # CLAUDE_NOTES.txt), so a minute-bar strategy fed Polygon-only bars is
    # really just re-evaluating yesterday's frozen close all day. Falls
    # through to Polygon on any failure, empty result, or when no live
    # source is available for this ticker's asset class.
    bars = None
    source_label = None
    live_data_account = _pick_live_data_source(ticker, broker_accounts, account_asset_classes)
    if live_data_account is not None:
        try:
            bars = live_data_account.get_live_bars(
                ticker=ticker, from_date=from_date, to_date=to_date,
                multiplier=control.multiplier, timespan=control.timespan,
            )
            if bars.empty:
                bars = None
            else:
                source_label = f"{live_data_account.nickname} (live)"
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{ticker}: live data fetch via {live_data_account.nickname} failed: {e}"
            bars = None

    if bars is None:
        try:
            bars = client.get_aggregates(
                ticker=ticker, from_date=from_date, to_date=to_date,
                multiplier=control.multiplier, timespan=control.timespan,
            )
            source_label = "Polygon"
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

    # Publish this cycle's read for the ticker regardless of hold/blocked/
    # executed outcome below — see current_signals.py. Conviction/levels here
    # are SEPARATE, display-only computations from entry_conviction further
    # down (which stays BUY-only, for sizing/attribution); both cost zero
    # extra Polygon calls since they're pure math over bars already fetched
    # above. recent_closes lets the mobile app draw a sparkline of what the
    # strategy is actually looking at without a second fetch of its own.
    snapshot_conviction = compute_conviction(strategy, bars, current) if signal.value != "hold" else None
    snapshot_levels = compute_levels(strategy, bars, current)
    ticker_asset_class = accounts_module.infer_asset_class(ticker)
    trading_accounts = [
        a for a in broker_accounts
        if ticker_asset_class in account_asset_classes.get(a.account_id, frozenset())
    ]
    market_open = (
        any(a.account_id not in market_closed_account_ids for a in trading_accounts)
        if trading_accounts else None
    )
    current_signals.record_signal(
        ticker=ticker, strategy_name=strategy_name, signal=signal.value,
        price=float(current.close), conviction=snapshot_conviction,
        bar_timestamp=current.timestamp.isoformat(), source=source_label,
        recent_closes=[float(c) for c in bars["close"].tail(RECENT_CLOSES_COUNT)],
        levels=snapshot_levels,
        market_open=market_open,
        trading_accounts=[a.nickname for a in trading_accounts],
    )

    if signal.value == "hold":
        return

    # Proactive step-out: never open NEW positions on a known risk-event day
    # (FOMC). Complements the reactive GARCH storm block below — this one fires
    # BEFORE the event moves the market. Sells fall through untouched.
    if control.block_event_days and signal.value == "buy":
        today_date = datetime.now(timezone.utc).date()
        reason = events.event_reason(today_date, ticker)
        if reason is not None:
            status.last_signal = f"{ticker}/{strategy_name}: BUY blocked — {reason} (event step-out)"
            save_status(status)
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

    # ticker_asset_class computed earlier, just before record_signal() above —
    # reused here rather than recomputed.

    for broker_account in broker_accounts:
        if broker_account.account_id in market_closed_account_ids:
            continue  # market shut for this account — never trade on stale bars

        if ticker_asset_class not in account_asset_classes.get(broker_account.account_id, frozenset()):
            continue  # e.g. an equity ticker against a forex-only IG account — see infer_asset_class

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

        if order_side is OrderSide.SELL:
            # Close what's actually held, not a fresh sizing calculation —
            # sizing_value/size_multiplier are an ENTRY (new-position) concept.
            # Recomputing "how many shares would today's sizing buy at today's
            # price" here only matched the held qty by coincidence (same price,
            # same size_multiplier as the original buy) and broke the moment
            # either changed: found live 2026-08-05 — DDOG failed to close 5x
            # in a row ("insufficient qty available") because a partial sell
            # had already shrunk the position, but each retry kept requesting
            # a freshly-computed ~1.09 shares against a ~0.003-share remainder.
            qty = existing_position.qty
        else:
            override = control.account_sizing_overrides.get(broker_account.account_id)
            effective_sizing_mode = sizing_mode
            effective_sizing_value = control.sizing_value
            if override is not None:
                # Small-account override: slides the %-of-equity rate from
                # slide_start_pct down toward the account's own global
                # sizing_value as equity grows, rather than a hard threshold
                # switch (the earlier design - found, after real backtesting,
                # to not line up with when the target rate actually clears a
                # real broker's minimum order size; see sliding_pct_equity's
                # own docstring). Checked fresh against a live snapshot every
                # entry, not decided once, so it naturally converges to plain
                # global sizing as the account grows - no manual switch-over,
                # and it can't drift out of sync with the global rate the way
                # a hardcoded threshold could, since target_pct is read live.
                try:
                    snapshot = broker_account.get_account_snapshot()
                except Exception as e:  # noqa: BLE001
                    status.last_error = f"{broker_account.nickname}: equity check for sizing override failed: {e}"
                    continue
                effective_sizing_mode = SizingMode.PCT_EQUITY
                effective_sizing_value = sliding_pct_equity(
                    snapshot.equity,
                    start_pct=override["slide_start_pct"],
                    target_pct=control.sizing_value,
                    floor_notional=override.get("slide_floor_notional", 1.0),
                )
            # Still scaled by size_multiplier either way, same as the global
            # path, so a GARCH storm-regime cut applies under the override too.
            effective_sizing_value *= size_multiplier
            try:
                qty = compute_qty_for_account(broker_account, current.close, effective_sizing_mode, effective_sizing_value)
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
            notifications.notify_order_rejected(
                ticker, order_side.value, ctx["account"].nickname, result.error or "unknown error"
            )
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
    # The external twin of the heartbeat above (#34). Deliberately fires before
    # every early return below, so it reports PROCESS LIVENESS, not armed-ness —
    # a disarmed-but-running bot is healthy and must not page anyone. Throttled
    # and exception-proof inside; it can never delay or block a trade.
    heartbeat.ping()

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
        # Union the primary's account_ids with every extra_targets group's
        # own account_ids — a forex extra target's IG account needs to be in
        # this same pool for its trades to happen at all. Safe when
        # extra_targets is empty (today's real state): identical to before.
        all_account_ids = set(control.account_ids) | {
            aid for group in control.extra_targets for aid in group.get("account_ids", [])
        }
        target_meta = [a for a in linked if a["id"] in all_account_ids]
        if not control.allow_live:
            target_meta = [a for a in target_meta if a["is_paper"]]
        if not target_meta:
            status.last_error = "no eligible target accounts (check allow_live / linked accounts)"
            save_status(status)
            return status
        broker_accounts = _get_broker_accounts([a["id"] for a in target_meta])
        # Per-account asset class (equity/crypto/forex), so a target ticker only
        # gets attempted against accounts that can actually trade its asset class
        # — see accounts.infer_asset_class's docstring for why: this is what lets
        # e.g. an IG (forex-only) account sit in control.account_ids permanently
        # alongside equity accounts without generating wasted API calls or error
        # noise every cycle an equity-only roster/manual config runs ("crosstalk").
        # Built from accounts.account_asset_class(), NOT the broker-level
        # BROKER_META[...]["asset_classes"] set directly — IBKR is the one broker
        # where that set has more than one member (a broker TYPE capability), but
        # any single LINKED IBKR account is still only ever equity OR forex, fixed
        # at link time. Using the broker-level set here would make every IBKR
        # account eligible for both, even one whose real IBKRBroker instance is
        # equity-only, silently defeating this exact guard.
        account_asset_classes = {
            a["id"]: frozenset({accounts_module.account_asset_class(a)}) for a in target_meta
        }
    except Exception as e:  # noqa: BLE001
        status.last_error = f"failed to build broker accounts: {e}"
        save_status(status)
        return status

    status.last_error = None

    # The event calendar is a static list (updated annually). If today is past
    # its coverage, is_event_day() silently returning False would mean "no
    # data", not "no event" — surface that instead of trading blind through
    # an unlisted FOMC meeting.
    if control.block_event_days and not events.calendar_covers(datetime.now(timezone.utc).date()):
        status.last_error = (
            f"event calendar only covers through {events.CALENDAR_COVERS_THROUGH} — "
            "update backtester/src/backtester/events.py with the Fed's newer meeting dates"
        )

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
            # Captured BEFORE check_and_update so the notification below fires
            # only on the actual trip (this cycle's transition into blocked),
            # not every subsequent cycle it stays blocked — it stays blocked
            # until a manual re-arm, which could be hours/days away.
            was_blocked = bool((account_risk.get_status(broker_account.account_id) or {}).get("blocked"))
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
                if not was_blocked:
                    notifications.notify_drawdown_blocked(broker_account.nickname, reason or "")

    if control.giveback_enabled:
        for broker_account in broker_accounts:
            if broker_account.account_id in blocked_account_ids:
                continue  # already blocked by the account-risk breaker above, no need to also check this
            was_blocked = bool((daily_pnl_guard.get_status(broker_account.account_id) or {}).get("blocked"))
            try:
                equity = broker_account.get_account_snapshot().equity
                blocked, reason = daily_pnl_guard.check_and_update(
                    broker_account.account_id, equity, control.giveback_pct
                )
            except Exception as e:  # noqa: BLE001
                status.last_error = f"{broker_account.nickname}: daily P&L guard check failed: {e}"
                continue
            if blocked:
                blocked_account_ids.add(broker_account.account_id)
                status.last_error = f"{broker_account.nickname}: daily P&L giveback limit reached — {reason}"
                if not was_blocked:
                    notifications.notify_giveback_blocked(broker_account.nickname, reason or "")

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
            blocked_account_ids, market_closed_account_ids, account_asset_classes,
        )

    save_status(status)
    return status


def _is_pid_alive(pid: int) -> bool:
    """Real OS-level process-liveness check, independent of anything either
    process wrote to disk — see diagnose_bot.py's identical helper, which
    exists for the same reason on the read-only diagnostic side."""
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _another_instance_alive() -> bool:
    """True if another auto-trader instance is genuinely alive AND actively
    looping right now. Requires BOTH signals to agree: (a) the recorded PID
    is a real, live OS process — not just a heartbeat that still "looks
    fresh" — and (b) that heartbeat is actually recent. Checking only the
    heartbeat is the bug this replaced: on 2026-08-04, a hard-killed process's
    last-written heartbeat still looked "fresh" for up to ~6 minutes
    afterward, blocking every restart attempt for the full window even though
    the PID was already gone (see CLAUDE_NOTES.txt). Checking only the PID
    isn't enough either — a hung-but-not-crashed process, or in the unlikely
    case the OS has already reused status.json's old PID for something
    unrelated, would incorrectly block a restart forever. Requiring both
    closes both gaps.

    Guards against running two instances at once (which would double every
    order), now that the trader can be auto-started on login as well as
    launched from the dashboard.
    """
    status = load_status()
    hb = status.last_heartbeat
    if not hb or not status.running or not status.pid or status.pid == os.getpid():
        return False
    if not _is_pid_alive(status.pid):
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(hb)).total_seconds()
    except ValueError:
        return False
    control = load_control()
    # a live loop beats at least once per poll interval; allow ~3 cycles of slack
    return age < max(300, 3 * control.poll_interval_seconds)


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
            control, readable = load_control_checked()
            status = run_cycle(status)
            # Only a GENUINE kill switch stops the process. An unreadable control
            # file also fails closed (run_cycle won't trade), but exiting on it
            # would mean a transient read problem silently kills the bot — the
            # exact "silence looks like no trades" failure mode we've been bitten
            # by before. Stay alive, keep the heartbeat, don't trade.
            if control.killed and readable:
                print("Kill switch engaged — stopping.")
                status.running = False
                save_status(status)
                break
            if not readable:
                print("Control file unreadable — not trading this cycle, staying alive.")
            time.sleep(max(control.poll_interval_seconds, 5))
    except KeyboardInterrupt:
        status.running = False
        save_status(status)
        print("Stopped by user.")


if __name__ == "__main__":
    main()
