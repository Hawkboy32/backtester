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
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from backtester import account_risk, accounts as accounts_module, daily_pnl_guard
from backtester import current_signals, events, heartbeat, live_trades, logging_setup, notifications, position_attribution, roster, source_stamp, version, volatility
from backtester.auto_trader_state import (
    AutoTraderStatus, load_control, load_control_checked, load_status, save_status,
)
from backtester.brokers.base import BrokerAccount, OrderSide
from backtester.brokers.ig import IGBroker
from backtester.data import PolygonClient, PolygonError
from backtester.engine import ENGINE_VERSION
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
from backtester.risk_presets import ticker_sizing_cap
from backtester.scan_db import record_scan
from backtester.scanner import run_scan
from backtester.strategies import STRATEGY_REGISTRY, build_strategy
from backtester.strategy import Bar
from backtester.universe import load_universe

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


def _resolve_targets(
    control, held_tickers: set[str] | None = None
) -> list[tuple[str, str, dict, bool, list[str] | None]]:
    """Returns (ticker, strategy_name, params, no_new_entries, account_ids)
    tuples to trade this cycle. Manual mode (primary): one entry per
    control.tickers, never blocked from new entries, params from
    control.manual_strategy_params (merged over STRATEGY_REGISTRY defaults
    inside build_strategy — empty dict here means "use the current
    default", same as always). Roster mode (primary): runs the cheap
    demotion-only pass (never promotes — that's a deliberate, separate
    dashboard action) and trades every active AND paused entry, so a paused
    entry can still exit an existing position, just never open a new one.

    Then, regardless of which primary branch ran, appends one target per
    ticker for every group in control.extra_targets — additional CONCURRENT
    manual-style targets (e.g. a forex account trading its own separately-
    tuned params alongside an equities roster primary). These are never
    blocked from new entries either.

    account_ids is None for primary targets (unrestricted — trade against
    every account in control.account_ids that matches the ticker's asset
    class, same as always) and a specific list for extra_targets groups,
    restricting that group to ONLY its own account_ids rather than every
    asset-class-matching account in the whole cycle's pool. Caller
    (_trade_target's broker_accounts argument) must actually scope by this
    — previously this field didn't exist at all and every extra_targets
    group silently traded on every matching account regardless of its own
    account_ids, found live 2026-08-25 (an "(PAPER)" target reached real
    AlpacaLive/IBKR Live accounts; nothing filled, but by luck, not by
    design — see CLAUDE_NOTES.txt "OPEN BUG 2026-08-25").
    """
    if not control.use_roster:
        primary = [
            (ticker, control.strategy_name, control.manual_strategy_params.get(control.strategy_name, {}), False, None)
            for ticker in control.tickers
        ]
    else:
        state = roster.load_roster()
        state = roster.apply_demotion_checks(state, live_trades.recent_performance, state.config)
        # Retire cap-demoted entries whose position has since closed, so
        # exit-only entries don't accumulate forever (see release_flat_paused).
        state, retired = roster.release_flat_paused(state, held_tickers or set())
        for label in retired:
            ticker, _, strategy_name = label.partition("/")
            roster.append_event(ticker, strategy_name, "retired", "position closed; leaving roster")
        roster.save_roster(state)

        primary = [
            (entry.ticker, entry.strategy_name, entry.params, entry.status == "paused", None)
            for entry in state.entries
            if entry.status in ("active", "paused")
        ]

    extra = [
        (ticker, group.get("strategy_name", ""), group.get("strategy_params", {}), False, group.get("account_ids", []))
        for group in control.extra_targets
        for ticker in group.get("tickers", [])
    ]
    return primary + extra


# 17:00 ET — an hour past NYSE close (16:00), a safety buffer against a scan
# firing during the last few minutes of trading if this check happens to run
# right at the boundary. Checked in America/New_York wall-clock time (DST-aware
# via ZoneInfo, same approach as VwapDriftPullbackStrategy's session filter)
# so the cutoff means the same real-world time year-round.
_ROSTER_CHECK_CUTOFF_HOUR = 17

# The only window in which an INDEX ticker is worth spending IG's scarce
# historical-price allowance on. See ig.py's get_live_bars docstring for the
# hard numbers: 10,000 points per WEEK, which an unconditional 120s poll would
# burn through in a single morning (~11,700/day). Opening Spike Fade - the
# only index strategy - can only ever act between the session open and the
# end of its reversal window (opening_minutes=5 then reversal_window_minutes=
# 20, so ~25 minutes), because it enters on exactly the first bar after the
# opening window and exits on reversion or the time-stop. Outside this window
# it structurally cannot open or close anything, so live data buys nothing
# there and the Polygon fallback is harmless.
#
# -2 covers a cycle landing just before the bell; +35 leaves a margin past the
# 25-minute worst case for a late/slow cycle. ~18 cycles x ~45 points is
# ~810/day, ~4,050/week worst case - comfortably inside the allowance.
_INDEX_LIVE_WINDOW_START_MIN = -2
_INDEX_LIVE_WINDOW_END_MIN = 35
_NY_TZ = ZoneInfo("America/New_York")
_US_OPEN_MINUTES = 9 * 60 + 30  # 09:30 ET


def _index_bars_window_open(now_utc: datetime | None = None) -> bool:
    """True only inside the window where an index ticker's live IG bars are
    actually actionable - see _INDEX_LIVE_WINDOW_* above for why this gate
    exists at all (IG's weekly allowance) and why this particular window is
    the right one.

    Weekday check only: US market holidays are NOT modelled here. A holiday
    costs at most one window's worth of allowance against a closed market
    (IG simply returns the previous session's tail), which is a far cheaper
    failure than wiring a holiday calendar into the data path.
    """
    now_ny = (now_utc or datetime.now(timezone.utc)).astimezone(_NY_TZ)
    if now_ny.weekday() >= 5:
        return False
    minutes_now = now_ny.hour * 60 + now_ny.minute
    return (
        _US_OPEN_MINUTES + _INDEX_LIVE_WINDOW_START_MIN
        <= minutes_now
        <= _US_OPEN_MINUTES + _INDEX_LIVE_WINDOW_END_MIN
    )


def _maybe_check_roster_gap(status: AutoTraderStatus, control) -> None:
    """Once per calendar day, after market close: if the adaptive roster has
    a gap (something paused, or fewer active entries than roster_size) OR a
    full review_cadence_days has elapsed since the last scan, run a REAL
    fresh scan and compute what evaluate_roster would recommend — see
    roster.compute_recommendation. The cadence leg catches a decaying edge
    via a fresh re-rank before live performance forces a demotion, not just
    reacting after a gap already opened. This is the one place in this file
    that deliberately runs an expensive scan; it's safe here specifically
    because it only ever fires after hours, never competing with intraday
    trading (evaluate_roster's own docstring says never to call it from
    inside the live poll loop during the day — this respects that by
    construction, not by working around it).

    NEVER applies anything. A real recommendation is saved via
    roster.save_pending() and pushed via one notification; the actual roster
    (what auto_trader.py trades) is untouched until a human approves it from
    the dashboard or mobile app. A failure anywhere in here is caught and
    logged to status.last_error — it must never be able to affect trading.
    """
    if not control.use_roster:
        return

    et_now = datetime.now(ZoneInfo("America/New_York"))
    today = et_now.date().isoformat()
    if status.last_roster_check_date == today:
        return
    if et_now.weekday() >= 5 or et_now.hour < _ROSTER_CHECK_CUTOFF_HOUR:
        return

    # Set this BEFORE attempting the scan, unconditionally — so a failure
    # below still counts as "checked today" and doesn't retry (and refail)
    # every poll cycle for the rest of the day.
    status.last_roster_check_date = today
    # ...and PERSIST it here, not just in memory. Found live 2026-09-02:
    # the assignment above only reaches disk via a later save_status(), so
    # when the scan below outran the watchdog's 660s staleness threshold the
    # process was killed before ever saving it. Every relaunch then re-read
    # the stale on-disk date, re-ran the same doomed scan, and was killed
    # again — seven relaunches across ~2.5 hours with the trader effectively
    # dead the whole time. Writing it now means a scan that still overruns
    # costs ONE cycle, never an unbounded loop.
    status.last_heartbeat = datetime.now(timezone.utc).isoformat()
    try:
        save_status(status)
    except Exception:  # noqa: BLE001
        pass  # a failed flag write must never stop the check itself

    try:
        state = roster.load_roster()
        active_count = sum(1 for e in state.entries if e.status == "active")
        has_gap = active_count < state.config.roster_size or any(e.status == "paused" for e in state.entries)

        cadence_due = True
        if status.last_roster_scan_date is not None:
            days_since_scan = (
                et_now.date() - date.fromisoformat(status.last_roster_scan_date)
            ).days
            cadence_due = days_since_scan >= state.config.review_cadence_days

        if not has_gap and not cadence_due:
            return

        status.last_roster_scan_date = today

        cfg = state.config
        tickers = sorted({
            t for universe_name in cfg.rescan_universes
            for t in load_universe(universe_name)["ticker"].tolist()
        })
        to_date = et_now.date()
        from_date = to_date - timedelta(days=cfg.rescan_window_days)

        scan_rows = run_scan(
            tickers=tickers,
            strategy_names=cfg.rescan_strategy_names,
            from_date=from_date.isoformat(),
            to_date=to_date.isoformat(),
            client=PolygonClient(),
            market_calendar="equity",
            # Keep proving liveness WHILE this runs. A full S&P 500 +
            # Nasdaq-100 rescan takes far longer than the watchdog's 660s
            # staleness threshold, so without this the watchdog kills a
            # perfectly healthy scan mid-flight and never lets it finish —
            # the 2026-09-02 crash loop above. Same reasoning as the
            # touch_heartbeat calls in run_cycle's own loops: a heartbeat
            # measures LIVENESS, not completion. Throttled internally to
            # one write per 30s, so this costs a handful of writes total.
            progress_callback=lambda _i, _total, _ticker: touch_heartbeat(status),
        )
        run_id = record_scan(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "universe": " + ".join(cfg.rescan_universes),
                "num_tickers": len(tickers),
                "from_date": from_date.isoformat(),
                "to_date": to_date.isoformat(),
                "multiplier": 1,
                "timespan": "minute",
                "strategy_names": cfg.rescan_strategy_names,
                "engine_version": ENGINE_VERSION,
            },
            scan_rows,
        )

        rec = roster.compute_recommendation(state, scan_rows, run_id, live_trades.recent_performance)
        if rec is not None:
            roster.save_pending(rec)
            notifications.notify_roster_recommendation_ready(rec.summary)
    except Exception as e:  # noqa: BLE001
        status.last_error = f"overnight roster check failed: {e}"


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
    this ticker's asset class and (b) exposes get_live_bars — AlpacaBroker
    (equities), CoinbaseBroker/KrakenBroker (crypto, added 2026-08-20 once
    Polygon's crypto data was found running up to ~21h stale in practice —
    Polygon is deliberately being pulled out of the live path project-wide
    from this point on, kept only for backtesting; CBAPI naturally wins over
    KRKAPI here since it's listed first in every crypto extra_targets entry
    AND is the only one of the two whose get_live_bars can page back far
    enough to cover a real strategy's lookback — see kraken.py's own
    docstring for the confirmed ~12h ceiling on Kraken's OHLC endpoint).
    Not account_asset_class-eligibility related
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

    # Index tickers (I:NDX): Alpaca rejects them outright ("invalid symbol:
    # I:NDX", seen live 2026-08-25) so the generic equity loop below would
    # pick Alpaca, fail, and fall through to Polygon - the last live target
    # still on Polygon after the 2026-08-20 migration. IG can serve them, but
    # only inside _index_bars_window_open()'s gate: its historical-price
    # allowance is far too scarce to poll all session (see that function and
    # ig.py's get_live_bars for the numbers). Outside the window we return
    # None deliberately - Polygon is a harmless fallback precisely when the
    # strategy structurally cannot act.
    if ticker.startswith("I:"):
        if not _index_bars_window_open():
            return None
        for broker_account in broker_accounts:
            if isinstance(broker_account, IGBroker):
                return broker_account
        return None

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

    # Prefer live data (Alpaca for equities, OANDA for forex, Coinbase/Kraken
    # for crypto — see _pick_live_data_source) over Polygon: this account's
    # Polygon plan has NO same-day intraday data at all for equities
    # (confirmed 2026-08-05), and its crypto data was separately found
    # running up to ~21h stale in practice (2026-08-20) despite no gap being
    # expected there either — so a minute-bar strategy fed Polygon-only bars
    # is really just re-evaluating a frozen, potentially very stale close all
    # day. Polygon is deliberately being kept for BACKTESTING ONLY from here
    # on, not as a live-signal source — every asset class now has a real
    # live path and Polygon should only ever be reached as a last-resort
    # fallback (e.g. mid-outage on every live source at once), not routine
    # behavior. Falls
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
    recent_bars = bars.tail(RECENT_CLOSES_COUNT)
    current_signals.record_signal(
        ticker=ticker, strategy_name=strategy_name, signal=signal.value,
        price=float(current.close), conviction=snapshot_conviction,
        bar_timestamp=current.timestamp.isoformat(), source=source_label,
        recent_closes=[float(c) for c in recent_bars["close"]],
        recent_opens=[float(o) for o in recent_bars["open"]],
        recent_highs=[float(h) for h in recent_bars["high"]],
        recent_lows=[float(l) for l in recent_bars["low"]],
        levels=snapshot_levels,
        market_open=market_open,
        trading_accounts=[a.nickname for a in trading_accounts],
    )

    if signal.value == "hold":
        return

    # Proactive step-out: never open a NEW position (either direction — see
    # account_position_modes) on a known risk-event day (FOMC). Complements
    # the reactive GARCH storm block below — this one fires BEFORE the event
    # moves the market. Applied per-account below, only when that account is
    # actually OPENING (a close always falls through untouched, regardless
    # of direction) — computed once here since it's the same for every
    # account this cycle.
    event_block_reason: str | None = None
    if control.block_event_days:
        today_date = datetime.now(timezone.utc).date()
        event_block_reason = events.event_reason(today_date, ticker)

    size_multiplier = 1.0
    storm_blocked = False
    regime_info = None
    if control.vol_target_enabled:
        regime_info = _get_regime(daily_client, ticker, control.vol_target_ann)
        if regime_info is not None:
            storm_blocked = regime_info.regime == "storm"
            size_multiplier = regime_info.size_multiplier

    status.last_signal = f"{ticker}/{strategy_name}: {signal.value.upper()} @ {current.close}"
    save_status(status)

    order_side = OrderSide.BUY if signal.value == "buy" else OrderSide.SELL
    sizing_mode = SizingMode(control.sizing_mode)
    # Score the entry signal's conviction once (#25) — pure math over bars
    # already fetched, safe to compute regardless of direction. Only
    # meaningful for an actual entry (open long OR open short — see
    # is_opening below); carried through attribution to live_trades on the
    # eventual close. Logged only — it does NOT influence sizing/gating.
    entry_conviction = compute_conviction(strategy, bars, current)

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
        has_long = existing_position is not None and existing_position.side == "long"
        has_short = existing_position is not None and existing_position.side == "short"

        # A submitted-but-unfilled order isn't a position yet, so the has_long/
        # has_short check alone can't stop us re-ordering the same ticker before
        # the first fill. Skip any ticker that already has an open order on this
        # account — covers slow fills during hours, not just the overnight case
        # above.
        try:
            if ticker in broker_account.get_open_order_tickers():
                continue
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: open-order check failed: {e}"
            continue

        # Same PositionMode concept engine.py already backtests (LONG_ONLY/
        # SHORT_ONLY/LONG_SHORT) — an account not in account_position_modes
        # defaults to long_only, reproducing every existing account's exact
        # prior behavior. Only meaningful for a broker whose submit_market_order
        # actually supports opening a short (OANDA always did; ig.py now does
        # too) — see account_position_modes' own docstring.
        position_mode = control.account_position_modes.get(broker_account.account_id, "long_only")
        can_open_long = position_mode in ("long_only", "long_short")
        can_open_short = position_mode in ("short_only", "long_short")

        # Four-way decision, mirroring engine.py's backtest logic exactly:
        # a BUY either opens a long (flat) or closes/covers a short (held);
        # a SELL either opens a short (flat) or closes a long (held). Already
        # holding the SAME direction the signal would open is a no-op (covers
        # the plain long-only case exactly as before: a BUY signal while
        # already long just does nothing, same as a stray SELL while flat).
        is_opening = False
        if order_side is OrderSide.BUY:
            if has_long:
                continue
            if not has_short:
                if not can_open_long or no_new_entries or broker_account.account_id in blocked_account_ids:
                    continue
                is_opening = True
        else:  # OrderSide.SELL
            if has_short:
                continue
            if not has_long:
                if not can_open_short or no_new_entries or broker_account.account_id in blocked_account_ids:
                    continue
                is_opening = True

        # Event-day/storm blocks only ever stop a NEW entry (either
        # direction) — a close always falls through untouched, matching the
        # pre-shorting "sells fall through untouched" behavior exactly.
        if is_opening and (event_block_reason is not None or storm_blocked):
            reason = event_block_reason or (
                f"GARCH storm regime (vol_pctile={regime_info.vol_pctile:.0f})" if regime_info else "storm regime"
            )
            status.last_signal = f"{ticker}/{strategy_name}: {order_side.value.upper()} blocked — {reason}"
            save_status(status)
            continue

        if not is_opening:
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
            # Per-ticker liquidity ceiling (2026-08-16) - layered on top of
            # whichever sizing got us here (global, override, or slide), a
            # pure safety MIN, never a way to size UP. Only touches
            # PCT_EQUITY, since the cap itself was measured in %-of-equity
            # terms and has no meaning against a fixed-dollar/fixed-share
            # target. Ticker not in the sweep -> None -> no cap, same as
            # today (uncapped is the status quo, not a new relaxation).
            if effective_sizing_mode == SizingMode.PCT_EQUITY:
                cap = ticker_sizing_cap(ticker, control.risk_preset)
                if cap is not None:
                    effective_sizing_value = min(effective_sizing_value, cap)
            try:
                qty = compute_qty_for_account(broker_account, current.close, effective_sizing_mode, effective_sizing_value, ticker)
            except Exception as e:  # noqa: BLE001
                status.last_error = f"{broker_account.nickname}: sizing failed: {e}"
                continue

        # Protective bracket prices, attached to the OPENING order only — a
        # closing order is already the exit, and handing a broker a stop on it
        # would leave a resting order against a position that no longer exists.
        take_profit_price = stop_loss_price = None
        if is_opening:
            take_profit_price, stop_loss_price = _bracket_prices(
                control, current.close, order_side
            )

        account_orders.append(AccountOrder(
            account=broker_account, qty=qty,
            take_profit_price=take_profit_price, stop_loss_price=stop_loss_price,
        ))
        # Sizing fields only meaningful when is_opening - captured HERE (per
        # account, inside this loop) because effective_sizing_mode/value are
        # loop-local and would silently hold the LAST account's numbers by
        # the time the results are processed below, otherwise misattributing
        # one account's sizing onto a different account's position.
        ctx = {"account": broker_account, "existing_position": existing_position, "is_opening": is_opening}
        if is_opening:
            ctx["sizing_mode"] = effective_sizing_mode.value
            ctx["sizing_value"] = effective_sizing_value
            ctx["dollars_committed"] = qty * current.close
        order_contexts.append(ctx)

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
        if ctx["is_opening"]:
            position_attribution.record_open(
                broker_account.account_id, ticker, strategy_name, conviction=entry_conviction,
                sizing_mode=ctx["sizing_mode"], sizing_value=ctx["sizing_value"],
                dollars_committed=ctx["dollars_committed"],
            )
            notifications.notify_trade_open(
                ticker, strategy_name, result.filled_qty or 0.0,
                broker_account.nickname, broker_account.is_paper,
            )
        else:
            attribution = position_attribution.pop_open(broker_account.account_id, ticker)
            if attribution is None:
                # No attribution — the position predates this flow, was opened
                # manually, or its record was pruned. This used to `continue`,
                # silently DISCARDING a real closed round trip: found
                # 2026-08-14 reconciling AlpacaLive, where an orphaned DDOG
                # close (-$0.07 real money) never reached live_trades and the
                # app's realised P&L was quietly wrong as a result.
                # Record it anyway under UNATTRIBUTED_STRATEGY: the P&L is
                # real and belongs in the books. The placeholder name keeps it
                # OUT of any real strategy's recent_performance() stats, so a
                # trade whose strategy we can't prove can't skew a demotion
                # decision either way.
                attribution = {
                    "strategy_name": live_trades.UNATTRIBUTED_STRATEGY,
                    "opened_at": "",
                    "conviction": None,
                }
            existing_position = ctx["existing_position"]
            exit_price = result.filled_avg_price or current.close
            qty = result.filled_qty or existing_position.qty
            # Sign qty by the CLOSED position's own direction, matching
            # engine.py's Trade convention (shares negative for a short) —
            # a short's economics are the OPPOSITE of a long's for the same
            # (exit - entry) sign: profit when price FALLS, not rises.
            signed_qty = qty if existing_position.side == "long" else -qty
            live_trades.record_realized_trade(
                account_id=broker_account.account_id,
                ticker=ticker,
                strategy_name=attribution["strategy_name"],
                is_paper=broker_account.is_paper,
                entry_time=attribution["opened_at"],
                entry_price=existing_position.avg_entry_price,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
                qty=signed_qty,
                conviction=attribution.get("conviction"),
            )
            pnl = (exit_price - existing_position.avg_entry_price) * signed_qty
            notifications.notify_trade_close(
                ticker, attribution["strategy_name"], qty, pnl,
                broker_account.nickname, broker_account.is_paper,
            )


def _bracket_prices(control, entry_price: float, order_side: OrderSide) -> tuple[float | None, float | None]:
    """(take_profit_price, stop_loss_price) for an OPENING order, or (None, None).

    Mirrors engine.BacktestEngine's protective exits so the live account is
    running the thing that was actually backtested — for a long the stop sits
    BELOW entry and the target above; for a short (SELL opens on a long_short
    account) both flip. Getting that flip wrong would submit a "stop" that is
    really a target, which most brokers accept without complaint.
    """
    if entry_price <= 0:
        return None, None
    is_long = order_side is OrderSide.BUY
    take_profit_price = stop_loss_price = None
    if control.take_profit_pct:
        take_profit_price = entry_price * (
            (1 + control.take_profit_pct) if is_long else (1 - control.take_profit_pct)
        )
    if control.stop_loss_pct:
        stop_loss_price = entry_price * (
            (1 - control.stop_loss_pct) if is_long else (1 + control.stop_loss_pct)
        )
    # Brokers reject sub-penny prices on US equities; round to the tick they
    # actually accept rather than having the whole opening order bounce.
    if take_profit_price is not None:
        take_profit_price = round(take_profit_price, 2)
    if stop_loss_price is not None:
        stop_loss_price = round(stop_loss_price, 2)
    return take_profit_price, stop_loss_price


def _flatten_before_close(control, broker_accounts, closing_soon_ids: set[str], status) -> None:
    """Close every open position on accounts whose session ends shortly.

    OFF by default and deliberately so: it lost money in all three backtest
    windows (see AutoTraderControl.flatten_before_close_minutes). It exists
    because "don't hold overnight" is a legitimate risk preference that should
    be available as a control, not because the evidence recommends it.

    Exits only — this can never open a position, so a bug here can only ever
    reduce exposure.
    """
    for broker_account in broker_accounts:
        if broker_account.account_id not in closing_soon_ids:
            continue
        try:
            positions = [p for p in broker_account.get_positions() if p.qty]
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: flatten position read failed: {e}"
            continue

        for position in positions:
            side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
            try:
                result = broker_account.submit_market_order(position.ticker, side, abs(position.qty))
            except Exception as e:  # noqa: BLE001
                status.last_error = f"{broker_account.nickname}: flatten {position.ticker} failed: {e}"
                continue
            if not result.success:
                notifications.notify_order_rejected(
                    position.ticker, side.value, broker_account.nickname,
                    result.error or "unknown error",
                )
                continue

            attribution = position_attribution.pop_open(broker_account.account_id, position.ticker)
            if attribution is None:
                attribution = {
                    "strategy_name": live_trades.UNATTRIBUTED_STRATEGY,
                    "opened_at": "", "conviction": None,
                }
            exit_price = result.filled_avg_price or position.current_price or position.avg_entry_price
            qty = result.filled_qty or abs(position.qty)
            signed_qty = qty if position.side == "long" else -qty
            live_trades.record_realized_trade(
                account_id=broker_account.account_id,
                ticker=position.ticker,
                strategy_name=attribution["strategy_name"],
                is_paper=broker_account.is_paper,
                entry_time=attribution["opened_at"],
                entry_price=position.avg_entry_price,
                exit_time=datetime.now(timezone.utc).isoformat(),
                exit_price=exit_price,
                qty=signed_qty,
                conviction=attribution.get("conviction"),
            )
            notifications.notify_trade_close(
                position.ticker, attribution["strategy_name"], qty,
                (exit_price - position.avg_entry_price) * signed_qty,
                broker_account.nickname, broker_account.is_paper,
            )


# Floor between mid-cycle heartbeat writes, so refreshing liveness across a
# 20+ ticker loop costs a handful of small writes rather than one per target.
_HEARTBEAT_MIN_WRITE_INTERVAL = 30.0
_last_heartbeat_write = 0.0


def touch_heartbeat(status: AutoTraderStatus) -> None:
    """Refresh the on-disk heartbeat MID-cycle, throttled.

    WHY (2026-08-26): last_heartbeat was stamped at the top of run_cycle but
    only written to disk by save_status() at the very END, so the file always
    carried a timestamp one whole cycle old. A healthy cycle takes ~60s and
    that was fine — but when IB Gateway went down overnight, every IBKR call
    burned its 20s timeout, cycles stretched past the watchdog's 360s
    staleness threshold, and a merely SLOW trader looked like a DEAD one. The
    watchdog dutifully relaunched it roughly every 11 minutes for two hours.
    (No duplicate ever ran - auto_trader's own singleton guard held - but the
    churn was pointless.)

    A heartbeat should measure LIVENESS, not cycle completion. Raising the
    watchdog's threshold instead would just blind it for longer; writing more
    often is the honest fix. Never raises: a failed heartbeat write must not
    be able to stop a trade.
    """
    global _last_heartbeat_write
    now = time.monotonic()
    if now - _last_heartbeat_write < _HEARTBEAT_MIN_WRITE_INTERVAL:
        return
    status.last_heartbeat = datetime.now(timezone.utc).isoformat()
    try:
        save_status(status)
        _last_heartbeat_write = now
    except Exception:  # noqa: BLE001
        pass


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

    # Not a trade, so deliberately not gated on trades_today's cap below — a
    # once-daily, after-hours check for whether the roster has a gap. See
    # _maybe_check_roster_gap's own docstring for why running a real scan here
    # is safe (only ever fires after market close).
    _maybe_check_roster_gap(status, control)

    # Daily trade cap. Deliberately NOT an early return any more: signal
    # publishing (current_signals.record_signal, which the mobile app and
    # dashboard both read) happens inside _trade_target below, so bailing out
    # here silently blacked out the app's live data for the rest of the day
    # once the cap was hit — found live 2026-08-12, the app fell back to
    # direct-Polygon fetches showing the PREVIOUS day's prices. The cap is a
    # limit on opening new risk, not a reason to stop observing the market.
    # Reuses the existing no_new_entries flag (same semantics a paused roster
    # entry already uses) so exits still work while the cap is in force —
    # being unable to CLOSE a position because of a counter is its own risk.
    cap_reached = status.trades_today >= control.max_trades_per_day
    if cap_reached:
        status.last_signal = (
            f"max trades/day ({control.max_trades_per_day}) reached — new entries blocked "
            "(exits and signal updates continue)"
        )
        save_status(status)

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
    closing_soon_account_ids: set[str] = set()
    next_open = None
    for broker_account in broker_accounts:
        touch_heartbeat(status)  # a wedged broker makes this loop the slow one
        try:
            clock = broker_account.get_market_clock()
        except Exception as e:  # noqa: BLE001
            status.last_error = f"{broker_account.nickname}: market clock check failed: {e}"
            continue
        if clock is not None and not clock["is_open"]:
            market_closed_account_ids.add(broker_account.account_id)
            next_open = clock.get("next_open")
        elif clock is not None and control.flatten_before_close_minutes:
            # Only meaningful for a broker that reports next_close; one that
            # doesn't simply never flattens rather than guessing a close time.
            next_close = clock.get("next_close")
            if next_close is not None:
                minutes_left = (next_close - datetime.now(timezone.utc)).total_seconds() / 60
                if 0 < minutes_left <= control.flatten_before_close_minutes:
                    closing_soon_account_ids.add(broker_account.account_id)

    if closing_soon_account_ids:
        _flatten_before_close(control, broker_accounts, closing_soon_account_ids, status)

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

    # Which tickers are actually held anywhere right now. Used only to retire
    # cap-demoted roster entries once they're genuinely flat (see
    # roster.release_flat_paused) — a combo that still holds something must
    # stay tradeable so it can be CLOSED. Best-effort: an unreachable broker
    # yields no tickers, which just defers the retirement a cycle rather than
    # wrongly declaring a position gone (the safe direction to fail).
    held_tickers: set[str] = set()
    for broker_account in broker_accounts:
        touch_heartbeat(status)  # position reads hit every broker; same reasoning
        try:
            held_tickers.update(p.ticker for p in broker_account.get_positions() if p.qty)
        except Exception:  # noqa: BLE001 — never let a position read stop the cycle
            continue

    targets = _resolve_targets(control, held_tickers)
    if not targets:
        status.last_signal = "roster empty — nothing to trade" if control.use_roster else status.last_signal
        save_status(status)
        return status

    for ticker, strategy_name, params, no_new_entries, target_account_ids in targets:
        # Prove liveness as the loop runs, not just when it finishes - see
        # touch_heartbeat. This is the loop that stretches when a broker is
        # unreachable, so it is exactly where the proof is needed.
        touch_heartbeat(status)
        control = load_control()  # re-check every target, not just once per cycle
        if control.killed or not control.enabled:
            status.last_signal = "stopped mid-cycle (killed or disabled)"
            save_status(status)
            return status
        # Re-checked per target (trades_today grows as this loop runs), and
        # folded into no_new_entries rather than breaking out — see the
        # cap_reached note above for why the loop must still run: every
        # remaining target needs its signal published even once the cap is
        # spent, otherwise the app goes dark for the rest of the day.
        cap_reached = status.trades_today >= control.max_trades_per_day

        # Scope to this target's own account_ids when it has one (extra_targets
        # groups) — a target with account_ids=None (primary) still gets the
        # full pool, unchanged from before. This is the actual enforcement;
        # see _resolve_targets' docstring for the incident that was missing it.
        target_broker_accounts = (
            broker_accounts if target_account_ids is None
            else [a for a in broker_accounts if a.account_id in target_account_ids]
        )

        _trade_target(
            ticker, strategy_name, params, no_new_entries or cap_reached,
            target_broker_accounts, client, daily_client, control, status,
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
                creationflags=subprocess.CREATE_NO_WINDOW,
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
    # As early as possible, before anything else could raise - see
    # logging_setup.py's own doc comment for why (a pythonw.exe process has
    # nowhere for print()/tracebacks to go otherwise, which is exactly how
    # the 2026-08-24 outage left zero trace of what killed it).
    logging_setup.configure("auto_trader")
    # Load .env by ABSOLUTE path (next to this file), not via cwd — so the bot
    # works identically whether launched from the dashboard, a terminal, or the
    # login auto-start task (which runs from an arbitrary working directory).
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if _another_instance_alive():
        print("Another auto-trader instance appears to be running (fresh heartbeat) — exiting to avoid double-trading.")
        return
    status = load_status()
    print(f"Auto-trader starting (pid={os.getpid()}, version={version.VERSION}).")
    # Records which source revision THIS process actually loaded, so the
    # dashboard can flag it as stale if the code on disk changes later
    # without a restart — see source_stamp.py for the two outages that
    # motivated it. Never raises.
    source_stamp.record_start("auto_trader")
    # Cycles since the last exception, purely for the backoff below - a
    # process that's erroring every single cycle should slow down rather than
    # hammer a broker/API that's clearly having a bad time, but one that's
    # mostly healthy shouldn't be punished by a single blip.
    consecutive_errors = 0
    try:
        while True:
            try:
                control, readable = load_control_checked()
                status = run_cycle(status)
                consecutive_errors = 0
            except Exception:  # noqa: BLE001
                # THE fix for the 2026-08-24 outage: this used to be
                # unguarded, so any exception here (a bad strategy param, a
                # broker API hiccup, anything) propagated straight out of
                # main() and killed the whole process - silently, since
                # pythonw.exe has no console for the traceback to appear on.
                # Logging it and continuing turns "the bot is dead until
                # someone notices" into "one cycle failed, logged, retried" -
                # the same "stay alive, keep the heartbeat" philosophy
                # already applied to an unreadable control file below.
                consecutive_errors += 1
                print(f"Cycle failed (consecutive={consecutive_errors}):")
                traceback.print_exc()
                status.last_heartbeat = datetime.now(timezone.utc).isoformat()
                try:
                    save_status(status)
                except Exception:  # noqa: BLE001
                    pass  # the heartbeat write itself failing must never stop the retry
                # Capped exponential-ish backoff: 5s, 10s, 20s, ... up to 5min,
                # so a persistent failure (e.g. a broker outage) doesn't spin
                # hot, but a one-off blip barely delays the next attempt.
                time.sleep(min(300, 5 * (2 ** min(consecutive_errors - 1, 6))))
                continue
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
