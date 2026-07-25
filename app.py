"""Password-protected Streamlit dashboard for the backtester.

Run with:  streamlit run app.py
First-time setup (creates your login):  python scripts/setup_auth.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit_authenticator as stauth
import yaml

from backtester import execution_log, keystore
from backtester.accounts import (
    BROKER_META,
    SUPPORTED_BROKERS,
    add_account,
    build_broker_accounts,
    list_accounts,
    remove_account,
)
from backtester.auto_trader_state import AutoTraderControl, load_control, load_status, save_control, trigger_kill_switch
from backtester.brokers.base import OrderSide
from backtester.brokers.ibkr import check_gateway_reachable
from backtester.data import PolygonClient, PolygonError
from backtester.engine import BacktestEngine
from backtester.execution import AccountOrder, SizingMode, compute_qty_for_account, execute_order_across_accounts
from backtester.memory_report import save_report
from backtester.metrics import compute_report
from backtester.ranking import aggregate_by_strategy, rank_combos
from backtester.saved_configs import delete_config, list_configs, load_config, save_config
from backtester.scan_db import (
    clear_history,
    query_all_time_top,
    query_latest_run_id,
    query_recent_runs,
    query_run_results,
    query_strategy_leaderboard,
    query_summary_counts,
    record_scan,
)
from backtester.scanner import run_scan
from backtester.strategies import STRATEGY_REGISTRY, build_strategy
from backtester.strategies.sma_crossover import SmaCrossoverStrategy
from backtester.universe import UNIVERSE_REGISTRY, load_universe
from backtester.walkforward import aggregate_walkforward, run_walkforward_scan
from backtester import account_risk, app_settings, live_trades, notifications, position_attribution, roster, volatility

PROJECT_ROOT = Path(__file__).resolve().parent
AUTO_TRADER_SCRIPT = PROJECT_ROOT / "auto_trader.py"
LIVE_ARM_PHRASE = "I ARM LIVE AUTO-TRADING"

st.set_page_config(page_title="Backtester Dashboard", layout="wide")

AUTH_CONFIG_PATH = Path(__file__).resolve().parent / "auth_config.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

COLOR_GAIN = "#00e676"
COLOR_LOSS = "#ff5252"
COLOR_NEUTRAL = "#8899a6"

TERMINAL_CSS = """
<style>
[data-testid="stMetricValue"] { font-family: 'Roboto Mono', 'Courier New', monospace; }
div[data-testid="stMetric"] {
    background-color: #121820;
    border: 1px solid #1e2a35;
    border-radius: 6px;
    padding: 10px 14px;
}
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    font-family: 'Roboto Mono', 'Courier New', monospace;
    background-color: #121820;
    border-radius: 4px 4px 0 0;
}
code, .stCode, .stCodeBlock { font-family: 'Roboto Mono', 'Courier New', monospace !important; }
.signal-feed {
    max-height: 420px;
    overflow-y: auto;
    background-color: #0b0f14;
    border: 1px solid #1e2a35;
    border-radius: 6px;
    padding: 8px;
}
.signal-feed-line {
    font-family: 'Roboto Mono', 'Courier New', monospace;
    font-size: 0.85rem;
    padding: 3px 8px;
    margin-bottom: 2px;
    border-left: 3px solid #8899a6;
    background-color: #121820;
    white-space: nowrap;
    overflow-x: auto;
}
</style>
"""


def signal_feed_line(ticker: str, strategy_name: str, total_return: float | None, sharpe_ratio: float | None, error: str | None) -> str:
    if error:
        color, arrow, detail = COLOR_NEUTRAL, "··", f"skip: {error}"
    elif total_return is not None and total_return > 0:
        color, arrow, detail = COLOR_GAIN, "▲", f"{total_return:+.2%}  sharpe {sharpe_ratio:.2f}"
    else:
        color, arrow, detail = COLOR_LOSS, "▼", f"{(total_return or 0):+.2%}  sharpe {(sharpe_ratio or 0):.2f}"
    return (
        f'<div class="signal-feed-line" style="border-left-color:{color};">'
        f'<span style="color:{color};font-weight:bold;">{arrow}</span> '
        f'<span style="color:#d7e2ea;">{ticker:6s}</span> '
        f'<span style="color:#8899a6;">{strategy_name}</span> '
        f'<span style="color:{color};">{detail}</span>'
        f"</div>"
    )


def _apply_pending_config_load(session_key: str, kind: str, date_keys: tuple[str, ...] = ()) -> None:
    """Must run before any widget for this tab is instantiated in this script pass —
    Streamlit forbids writing session_state for a key whose widget already rendered
    this run, so a queued load is applied here, at the top of the tab function,
    rather than inside the button handler that queued it.
    """
    pending_name = st.session_state.pop(session_key, None)
    if not pending_name:
        return
    cfg = load_config(kind, pending_name)
    if not cfg:
        return
    for k, v in cfg.items():
        if k in date_keys and isinstance(v, str):
            v = datetime.strptime(v, "%Y-%m-%d").date()
        st.session_state[k] = v


def _render_saved_config_ui(
    kind: str, widget_keys: list[str], pending_session_key: str, date_keys: tuple[str, ...] = ()
) -> None:
    with st.expander("Saved configurations"):
        names = list_configs(kind)

        col1, col2 = st.columns([3, 1])
        with col1:
            selected = st.selectbox(
                "Load a saved configuration", options=["(none)"] + names, key=f"{kind}_load_select"
            )
        with col2:
            st.write("")
            if st.button("Load", key=f"{kind}_load_btn") and selected != "(none)":
                st.session_state[pending_session_key] = selected
                st.rerun()

        col3, col4 = st.columns([3, 1])
        with col3:
            new_name = st.text_input("Save current settings as", key=f"{kind}_save_name")
        with col4:
            st.write("")
            if st.button("Save configuration", key=f"{kind}_save_btn") and new_name:
                params = {}
                for k in widget_keys:
                    v = st.session_state.get(k)
                    if k in date_keys and hasattr(v, "isoformat"):
                        v = v.isoformat()
                    params[k] = v
                save_config(kind, new_name, params)
                st.success(f"Saved as '{new_name}'.")
                st.rerun()

        if names:
            col5, col6 = st.columns([3, 1])
            with col5:
                del_name = st.selectbox(
                    "Delete a saved configuration", options=["(none)"] + names, key=f"{kind}_delete_select"
                )
            with col6:
                st.write("")
                if st.button("Delete", key=f"{kind}_delete_btn") and del_name != "(none)":
                    delete_config(kind, del_name)
                    st.success(f"Deleted '{del_name}'.")
                    st.rerun()


def require_auth() -> tuple[stauth.Authenticate, str]:
    if not AUTH_CONFIG_PATH.exists():
        st.error(
            "No login configured yet. Run this in a terminal first, then reload:\n\n"
            "    python scripts/setup_auth.py"
        )
        st.stop()

    config = yaml.safe_load(AUTH_CONFIG_PATH.read_text())
    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    authenticator.login("main")

    auth_status = st.session_state.get("authentication_status")
    if auth_status is False:
        st.error("Username or password is incorrect.")
        st.stop()
    if auth_status is None:
        st.warning("Please log in to continue.")
        st.stop()

    username = st.session_state.get("username", "")
    return authenticator, username


def render_settings_page() -> None:
    """One home for all configuration: API keys, phone notifications, the
    execution-cost defaults new Backtest/Scanner runs start from, and the
    account-level max-drawdown circuit breaker. Consolidated here so settings
    stop being scattered across the API Keys tab, the Backtest/Scanner sidebars,
    and the Auto Trading tab."""
    st.subheader("Settings")
    st.caption("API keys, notifications, execution-cost defaults, and the account-risk limit — all in one place.")

    st.markdown("### API keys")
    st.caption(
        "Stored locally in this project's .env file. Never sent anywhere except "
        "to the API providers themselves when the backtester runs."
    )

    current = keystore.list_keys()
    for env_name, label in keystore.KNOWN_KEYS.items():
        col1, col2 = st.columns([2, 3])
        with col1:
            st.text(f"{label}")
            st.caption(f"Current: {keystore.mask(current.get(env_name))}")
        with col2:
            with st.form(key=f"form_{env_name}", clear_on_submit=True):
                new_value = st.text_input(
                    f"New {label} value", type="password", key=f"input_{env_name}"
                )
                submitted = st.form_submit_button("Save")
                if submitted and new_value:
                    keystore.save_key(env_name, new_value)
                    st.success(f"{label} updated.")
                    st.rerun()

    st.divider()
    st.markdown("### Phone notifications")
    st.caption(
        "Get a push to your phone each time the auto-trader opens or closes a trade. "
        "Uses ntfy (free): install the **ntfy** app (iOS/Android), then in the app "
        "subscribe to the exact topic name you set below. No account or password needed — "
        "so pick a long, unguessable topic name (anyone who knows it can see your pings)."
    )
    notif_cfg = notifications.load_config()
    notif_enabled = st.checkbox(
        "Enable trade notifications", value=notif_cfg.get("enabled", False), key="notif_enabled"
    )
    notif_topic = st.text_input(
        "ntfy topic name",
        value=notif_cfg.get("ntfy_topic", ""),
        key="notif_topic",
        help="Any unique string, e.g. mybot-aa39f1c2b7. Subscribe to this exact name in the ntfy app.",
    )
    ncol1, ncol2 = st.columns(2)
    with ncol1:
        if st.button("Save notification settings", key="notif_save"):
            notifications.save_config(
                {"enabled": notif_enabled, "provider": "ntfy", "ntfy_topic": notif_topic.strip()}
            )
            st.success("Notification settings saved.")
    with ncol2:
        if st.button("Send test notification", key="notif_test"):
            ok = notifications.notify(
                "Test notification",
                "If you can read this on your phone, trade alerts are working.",
                config={"enabled": True, "provider": "ntfy", "ntfy_topic": notif_topic.strip()},
            )
            if ok:
                st.success("Sent — check your phone. If nothing arrives, confirm the app is subscribed to that exact topic.")
            elif not notif_topic.strip():
                st.error("Enter a topic name first.")
            else:
                st.error("Send failed — check your internet connection and the topic name.")

    st.divider()
    st.markdown("### Execution-cost defaults")
    st.caption(
        "The commission and slippage that new Backtest and Scanner runs start from. Defaults "
        "model Alpaca: $0 commission on US stocks/ETFs, with spread-crossing + sub-bp sell-side "
        "regulatory fees captured as ~2 bps slippage per side. You can still override these per "
        "run in the Backtest/Scanner sidebars."
    )
    cost_cfg = app_settings.load_settings()
    # Seed the widgets from the persisted defaults before they render, so the preset
    # buttons below can set them via session_state without the value=+key warning.
    if "settings_commission" not in st.session_state:
        st.session_state["settings_commission"] = cost_cfg["commission"]
    if "settings_slippage" not in st.session_state:
        st.session_state["settings_slippage"] = cost_cfg["slippage_bps"]

    def _fill_cost_preset(slippage_bps: float) -> None:
        # Fill the Settings inputs; commission stays $0 either way (Alpaca).
        st.session_state["settings_commission"] = 0.0
        st.session_state["settings_slippage"] = slippage_bps

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.button(
            f"Large cap ({LARGE_CAP_SLIPPAGE_BPS:.0f} bps)", key="settings_cost_large",
            width="stretch", on_click=_fill_cost_preset, args=(LARGE_CAP_SLIPPAGE_BPS,),
        )
    with pcol2:
        st.button(
            f"Small cap ({SMALL_CAP_SLIPPAGE_BPS:.0f} bps)", key="settings_cost_small",
            width="stretch", on_click=_fill_cost_preset, args=(SMALL_CAP_SLIPPAGE_BPS,),
        )
    st.number_input("Default commission per trade ($)", min_value=0.0, step=0.5, key="settings_commission")
    st.number_input("Default slippage (bps)", min_value=0.0, step=0.5, key="settings_slippage")
    if st.button("Save execution-cost defaults", key="settings_save_costs"):
        commission = float(st.session_state["settings_commission"])
        slippage = float(st.session_state["settings_slippage"])
        app_settings.save_settings({"commission": commission, "slippage_bps": slippage})
        # Push into the live Backtest/Scanner widgets so the change applies immediately,
        # not just on a fresh session (their keys may already hold an old value).
        st.session_state["bt_commission"] = commission
        st.session_state["scan_comm"] = commission
        st.session_state["bt_slippage"] = slippage
        st.session_state["scan_slip"] = slippage
        st.success("Execution-cost defaults saved.")

    st.divider()
    st.markdown("### Account risk limit")
    st.caption(
        "A per-account hard stop, independent of Manual/Adaptive mode. Once an account's equity "
        "drops this far below its own peak since being watched, new entries are blocked (existing "
        "positions can still be closed). It stays blocked — even if equity recovers — until you "
        "manually re-arm it below."
    )
    control = load_control()
    risk_enabled = st.checkbox(
        "Enable account-level max-drawdown circuit breaker. Off by default.",
        value=control.max_drawdown_enabled, key="settings_max_dd_enabled",
    )
    risk_pct = st.number_input(
        "Max drawdown from peak equity (%)", min_value=1.0, max_value=100.0,
        value=control.max_drawdown_pct, step=1.0, key="settings_max_dd_pct", disabled=not risk_enabled,
    )
    if st.button("Save account-risk limit", key="settings_save_risk"):
        # Load-mutate-save so only the two drawdown fields change — never clobber the
        # rest of the auto-trader control (which the Auto Trading tab also writes).
        fresh = load_control()
        fresh.max_drawdown_enabled = risk_enabled
        fresh.max_drawdown_pct = risk_pct
        save_control(fresh)
        st.success("Account-risk limit saved.")
        st.rerun()

    linked = list_accounts()
    watched = [a for a in linked if a["id"] in control.account_ids]
    blocked_any = False
    for account in watched:
        risk_status = account_risk.get_status(account["id"])
        if risk_status and risk_status.get("blocked"):
            blocked_any = True
            rcol1, rcol2 = st.columns([3, 1])
            with rcol1:
                st.error(f"{account['nickname']}: risk-blocked — {risk_status['reason']}")
            with rcol2:
                if st.button("Re-arm", key=f"settings_rearm_{account['id']}"):
                    try:
                        broker_accounts = build_broker_accounts([account["id"]])
                        equity = broker_accounts[0].get_account_snapshot().equity
                        account_risk.reset_breach(account["id"], equity)
                        st.success(f"{account['nickname']} re-armed at current equity ${equity:,.2f}.")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Could not re-arm: {e}")
    if control.account_ids and not blocked_any:
        st.caption("No target accounts are currently risk-blocked.")


BACKTEST_CONFIG_KEYS = [
    "bt_ticker", "bt_from", "bt_to", "bt_mult", "bt_timespan", "bt_fast", "bt_slow",
    "bt_cash", "bt_commission", "bt_slippage", "bt_compare_buy_hold",
    "bt_vol_target", "bt_target_vol",
]
BACKTEST_DATE_KEYS = ("bt_from", "bt_to")


@st.cache_data(ttl="1h")
def _known_tickers() -> dict[str, str]:
    """Ticker -> company name across every universe CSV, for the Backtest
    ticker dropdown. Cached: the constituent CSVs only change when their
    refresh scripts are run."""
    known: dict[str, str] = {}
    for universe_name in UNIVERSE_REGISTRY:
        try:
            df = load_universe(universe_name)
        except FileNotFoundError:
            continue
        for _, row in df.iterrows():
            name = row["name"] if "name" in row and isinstance(row["name"], str) else ""
            known.setdefault(row["ticker"], name)
    return known


def _friendly_broker_error(error: str | None) -> str:
    """Broker rejections often arrive as raw JSON — surface the human-readable
    message field when there is one, the raw text otherwise."""
    if not error:
        return "unknown error"
    try:
        parsed = json.loads(error)
        if isinstance(parsed, dict) and parsed.get("message"):
            return str(parsed["message"])
    except (json.JSONDecodeError, TypeError):
        pass
    return error


# Liquidity-based slippage presets, offered on the Settings page's execution-cost
# defaults section. Commission stays $0 either way (Alpaca); only the spread differs.
LARGE_CAP_SLIPPAGE_BPS = 2.0   # liquid S&P/Nasdaq names: ~1-2 bps half-spread
SMALL_CAP_SLIPPAGE_BPS = 10.0  # thinner names: wider spreads, conservative default


def render_backtest_tab() -> tuple[dict, "st.delta_generator.DeltaGenerator"] | None:
    """Renders the Backtest tab UI. If a run was requested and passes
    validation, returns (params, results_container) for main() to execute
    AFTER every tab has rendered — running it inline here would block the
    script mid-render, leaving all the tabs after this one (Scanner, etc.)
    stuck looking half-loaded for the duration of the backtest.
    """
    _apply_pending_config_load("_bt_pending_load", "backtest", BACKTEST_DATE_KEYS)

    st.subheader("Strategy & backtest parameters")
    _render_saved_config_ui("backtest", BACKTEST_CONFIG_KEYS, "_bt_pending_load", BACKTEST_DATE_KEYS)

    with st.sidebar:
        st.header("Parameters")
        known = _known_tickers()
        ticker_options = sorted(known)
        # A saved config or a hand-typed symbol from an earlier run may not be
        # in the universe lists — keep it selectable rather than erroring.
        current_ticker = st.session_state.get("bt_ticker")
        if current_ticker and current_ticker not in known:
            ticker_options = [current_ticker] + ticker_options
        ticker_choice = st.selectbox(
            "Ticker (type to search, or enter any symbol not listed)",
            options=ticker_options,
            index=ticker_options.index("AAPL") if "AAPL" in ticker_options else 0,
            format_func=lambda t: f"{t} — {known[t]}" if known.get(t) else t,
            key="bt_ticker",
            accept_new_options=True,
        )
        ticker = (ticker_choice or "AAPL").upper().strip()
        col1, col2 = st.columns(2)
        with col1:
            from_date = st.date_input(
                "From", value=pd.Timestamp.today() - pd.Timedelta(days=180), key="bt_from"
            )
        with col2:
            to_date = st.date_input("To", value=pd.Timestamp.today(), key="bt_to")

        st.divider()
        st.caption("Data granularity")
        multiplier = st.number_input("Bar multiplier", min_value=1, value=1, step=1, key="bt_mult")
        timespan = st.selectbox("Bar unit", ["minute", "hour", "day"], index=0, key="bt_timespan")

        st.divider()
        st.caption("SMA crossover strategy")
        fast_window = st.number_input("Fast window", min_value=1, value=20, step=1, key="bt_fast")
        slow_window = st.number_input("Slow window", min_value=2, value=50, step=1, key="bt_slow")

        st.divider()
        st.caption(
            "Execution assumptions — defaults model Alpaca: $0 commission on US "
            "stocks/ETFs, with spread-crossing + sub-bp sell-side regulatory fees "
            "captured as ~2 bps slippage per side. Set the default in Settings; adjust here per run."
        )
        _bt_costs = app_settings.load_settings()
        starting_cash = st.number_input(
            "Starting cash ($)", min_value=1.0, value=100_000.0, step=1000.0, key="bt_cash"
        )
        commission = st.number_input(
            "Commission per trade ($)", min_value=0.0, value=_bt_costs["commission"], step=0.5, key="bt_commission"
        )
        slippage_bps = st.number_input(
            "Slippage (bps)", min_value=0.0, value=_bt_costs["slippage_bps"], step=0.5, key="bt_slippage"
        )
        compare_buy_hold = st.checkbox(
            "Compare to buy-and-hold", value=True, key="bt_compare_buy_hold"
        )

        st.divider()
        st.caption("GARCH volatility filter + sizing")
        vol_target_enabled = st.checkbox(
            "Enable: block entries in high-vol (\"storm\") regime, scale position "
            "size by forecast vol otherwise. Needs ~2 years of daily history for "
            "this ticker; off by default so it doesn't change existing results.",
            value=False, key="bt_vol_target",
        )
        target_vol_ann = st.number_input(
            "Target annualized vol (%)", min_value=1.0, value=20.0, step=1.0,
            key="bt_target_vol", disabled=not vol_target_enabled,
        )

        run_clicked = st.button("Run backtest", type="primary", use_container_width=True)

    if not run_clicked:
        st.info("Set parameters in the sidebar and click **Run backtest**.")
        return None

    if fast_window >= slow_window:
        st.error("Fast window must be smaller than slow window.")
        return None

    if not keystore.get_key("POLYGON_API_KEY"):
        st.error("No Polygon API key set. Add one in the **API Keys** tab first.")
        return None

    params = {
        "ticker": ticker,
        "from_date": from_date,
        "to_date": to_date,
        "multiplier": multiplier,
        "timespan": timespan,
        "fast_window": fast_window,
        "slow_window": slow_window,
        "starting_cash": starting_cash,
        "commission": commission,
        "slippage_bps": slippage_bps,
        "compare_buy_hold": compare_buy_hold,
        "vol_target_enabled": vol_target_enabled,
        "target_vol_ann": target_vol_ann,
    }
    return params, st.container()


def execute_backtest(params: dict, container) -> None:
    """The actual backtest run, extracted from render_backtest_tab so main()
    can defer it until after every tab has rendered (see that docstring).
    Writes all progress and results into the Backtest tab's own container.
    """
    with container:
        _execute_backtest_body(params)


def _execute_backtest_body(params: dict) -> None:
    ticker = params["ticker"]
    from_date = params["from_date"]
    to_date = params["to_date"]
    multiplier = params["multiplier"]
    timespan = params["timespan"]
    fast_window = params["fast_window"]
    slow_window = params["slow_window"]
    starting_cash = params["starting_cash"]
    commission = params["commission"]
    slippage_bps = params["slippage_bps"]
    compare_buy_hold = params["compare_buy_hold"]
    vol_target_enabled = params["vol_target_enabled"]
    target_vol_ann = params["target_vol_ann"]

    with st.status("Running backtest...", expanded=True) as status:
        try:
            st.write(f"Fetching {timespan} bars for {ticker} from {from_date} to {to_date}...")
            client = PolygonClient()
            bars = client.get_aggregates(
                ticker=ticker,
                from_date=str(from_date),
                to_date=str(to_date),
                multiplier=int(multiplier),
                timespan=timespan,
            )
            st.write(f"Fetched {len(bars)} bars.")

            if bars.empty:
                status.update(label="No data returned", state="error")
                st.error("No bars returned for that ticker/date range.")
                return

            regime_by_date = None
            latest_regime = None
            if vol_target_enabled:
                st.write("Fetching daily history for the GARCH volatility regime...")
                try:
                    daily_from = (pd.Timestamp(from_date) - pd.Timedelta(days=1100)).date().isoformat()
                    daily_bars = client.get_aggregates(
                        ticker=ticker, from_date=daily_from, to_date=str(to_date), multiplier=1, timespan="day",
                    )
                    regime_table = volatility.compute_regime_table(daily_bars, target_vol_ann=target_vol_ann)
                    regime_by_date = volatility.regime_by_date(regime_table)
                    latest_regime = volatility.latest_regime_info(regime_table)
                except volatility.InsufficientHistoryError as e:
                    st.warning(f"Not enough daily history for a GARCH regime on {ticker}: {e} Running without it.")
                except PolygonError as e:
                    st.warning(f"Daily history fetch failed for the GARCH regime: {e} Running without it.")

            st.write("Running strategy over bars...")
            strategy = SmaCrossoverStrategy(fast_window=int(fast_window), slow_window=int(slow_window))
            engine = BacktestEngine(
                starting_cash=starting_cash,
                commission_per_trade=commission,
                slippage_bps=slippage_bps,
                regime_by_date=regime_by_date,
            )
            result = engine.run(bars, strategy)

            st.write("Computing performance metrics...")
            report = compute_report(result.equity_curve, result.trades)

            status.update(label="Done", state="complete")
        except PolygonError as e:
            status.update(label="Failed", state="error")
            st.error(f"Polygon API error: {e}")
            return
        except Exception as e:  # noqa: BLE001
            status.update(label="Failed", state="error")
            st.error(f"Backtest failed: {e}")
            return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total return", f"{report.total_return:.2%}")
    m2.metric("CAGR", f"{report.cagr:.2%}")
    m3.metric("Max drawdown", f"{report.max_drawdown:.2%}")
    m4.metric("Sharpe ratio", f"{report.sharpe_ratio:.2f}")
    m5.metric("Trades / win rate", f"{report.num_trades} / {report.win_rate:.0%}")

    if vol_target_enabled and latest_regime is not None:
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Current vol regime", latest_regime.regime.upper())
        rc2.metric("Forecast vol (annualized)", f"{latest_regime.fcast_vol_ann_pct:.1f}%")
        rc3.metric("Position size multiplier", f"{latest_regime.size_multiplier:.2f}x")
        st.caption(volatility.HONESTY_NOTE)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, mode="lines", name="Strategy")
    )
    if compare_buy_hold:
        buy_hold_equity = bars["close"] / bars["close"].iloc[0] * starting_cash
        fig.add_trace(
            go.Scatter(
                x=buy_hold_equity.index, y=buy_hold_equity.values, mode="lines", name="Buy & hold",
                line=dict(dash="dot"),
            )
        )
    fig.update_layout(title="Equity curve", xaxis_title="Time", yaxis_title="Equity ($)", height=400)
    st.plotly_chart(fig, use_container_width=True)

    if result.trades:
        trade_rows = [
            {
                "entry_time": t.entry_time,
                "entry_price": round(t.entry_price, 2),
                "exit_time": t.exit_time,
                "exit_price": round(t.exit_price, 2) if t.exit_price is not None else None,
                "shares": round(t.shares, 4),
                "pnl": round(t.pnl, 2) if t.pnl is not None else None,
                "conviction": round(t.conviction, 3) if t.conviction is not None else None,
            }
            for t in result.trades
        ]
        st.subheader("Trade log")
        st.dataframe(pd.DataFrame(trade_rows), use_container_width=True)
    else:
        st.info("No trades were executed for these parameters.")


SCANNER_CONFIG_KEYS = [
    "scan_universe", "scan_max_tickers", "scan_strategies", "scan_from", "scan_to",
    "scan_mult", "scan_timespan", "scan_walkforward", "scan_num_folds", "scan_rpm",
    "scan_max_workers", "scan_cash", "scan_comm", "scan_slip",
    "scan_vol_target", "scan_target_vol",
]
SCANNER_DATE_KEYS = ("scan_from", "scan_to")


def render_scanner_tab() -> None:
    _apply_pending_config_load("_scan_pending_load", "scanner", SCANNER_DATE_KEYS)

    st.subheader("Scan strategies across an index")
    st.caption(
        "Runs every selected strategy against every selected ticker's Polygon bars, "
        "ranks the results by a composite score, and writes a bot_memory.txt summary."
    )
    _render_saved_config_ui("scanner", SCANNER_CONFIG_KEYS, "_scan_pending_load", SCANNER_DATE_KEYS)
    st.caption(
        "⚠ Both universes below are today's constituent list applied retroactively over "
        "the historical window — this overstates performance somewhat, since companies "
        "removed/delisted from the index along the way aren't included (survivorship bias)."
    )

    universe_name = st.selectbox("Universe", options=list(UNIVERSE_REGISTRY.keys()), key="scan_universe")
    try:
        universe_df = load_universe(universe_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    col1, col2 = st.columns(2)
    with col1:
        max_tickers = st.slider(
            f"Number of {universe_name} tickers to scan (alphabetical subset)",
            min_value=1,
            max_value=len(universe_df),
            value=min(25, len(universe_df)),
            step=1,
            key="scan_max_tickers",
        )
        strategy_names = st.multiselect(
            "Strategies to test",
            options=list(STRATEGY_REGISTRY.keys()),
            default=list(STRATEGY_REGISTRY.keys()),
            key="scan_strategies",
        )
    with col2:
        from_date = st.date_input(
            "From", value=pd.Timestamp.today() - pd.Timedelta(days=30), key="scan_from"
        )
        to_date = st.date_input("To", value=pd.Timestamp.today(), key="scan_to")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            multiplier = st.number_input("Bar multiplier", min_value=1, value=1, step=1, key="scan_mult")
        with gcol2:
            timespan = st.selectbox("Bar unit", ["minute", "hour", "day"], index=0, key="scan_timespan")

    walkforward_enabled = st.checkbox(
        "Walk-forward validation: split the date range into sequential folds and check "
        "whether performance holds up across periods, instead of trusting one window",
        key="scan_walkforward",
    )
    num_folds = 1
    if walkforward_enabled:
        num_folds = st.slider("Number of folds", min_value=2, max_value=8, value=3, key="scan_num_folds")

    with st.expander("Advanced: rate limit, concurrency, execution assumptions"):
        requests_per_minute = st.number_input(
            "Polygon requests/minute (conservative default since your plan tier is unconfirmed "
            "— raise this only if you know your plan supports more)",
            min_value=1,
            value=5,
            step=1,
            key="scan_rpm",
        )
        max_workers = st.number_input(
            "Concurrent strategy workers per ticker", min_value=1, value=4, step=1, key="scan_max_workers"
        )
        starting_cash = st.number_input(
            "Starting cash ($)", min_value=1.0, value=100_000.0, step=1000.0, key="scan_cash"
        )
        _scan_costs = app_settings.load_settings()
        commission = st.number_input(
            "Commission per trade ($) — default models Alpaca's $0 US stock/ETF commission",
            min_value=0.0, value=_scan_costs["commission"], step=0.5, key="scan_comm",
        )
        slippage_bps = st.number_input(
            "Slippage (bps) — default ~2 bps/side covers spread-crossing + sell-side regulatory fees on Alpaca",
            min_value=0.0, value=_scan_costs["slippage_bps"], step=0.5, key="scan_slip",
        )

    st.caption("GARCH volatility filter + sizing")
    vol_target_enabled = st.checkbox(
        "Enable: block entries in high-vol (\"storm\") regime, scale position size by "
        "forecast vol otherwise. Needs a separate daily-bar fetch per ticker (~2 years of "
        "history), roughly doubling this scan's API calls. Off by default so it doesn't "
        "change existing scan results.",
        value=False, key="scan_vol_target",
    )
    target_vol_ann = st.number_input(
        "Target annualized vol (%)", min_value=1.0, value=20.0, step=1.0,
        key="scan_target_vol", disabled=not vol_target_enabled,
    )

    if not strategy_names:
        st.warning("Select at least one strategy.")
        return

    calls_per_ticker = 2 if vol_target_enabled else 1
    est_minutes = max_tickers * calls_per_ticker / requests_per_minute
    st.caption(
        f"Estimated minimum runtime: ~{max_tickers * calls_per_ticker} API calls "
        f"({calls_per_ticker}/ticker) at {requests_per_minute}/min "
        f"≈ {est_minutes:.1f} minutes (more if any ticker needs multiple pages of data). "
        f"This runs synchronously in this browser tab — keep it open until it finishes. "
        f"Progress is checkpointed, so an interrupted scan with identical settings will resume."
    )

    config_key = (
        f"{universe_name}|{max_tickers}|{sorted(strategy_names)}|{from_date}|{to_date}|"
        f"{multiplier}|{timespan}|{vol_target_enabled}|{target_vol_ann}"
    )
    checkpoint_hash = hashlib.sha256(config_key.encode()).hexdigest()[:16]
    results_dir = RESULTS_DIR / checkpoint_hash
    checkpoint_path = results_dir / "scan_checkpoint.jsonl"

    if checkpoint_path.exists():
        st.info("Found a checkpoint for this exact configuration — re-running will skip completed tickers.")

    run_clicked = st.button("Run scan", type="primary")

    if not run_clicked:
        return

    if not keystore.get_key("POLYGON_API_KEY"):
        st.error("No Polygon API key set. Add one in the **API Keys** tab first.")
        return

    tickers = universe_df["ticker"].head(max_tickers).tolist()
    client = PolygonClient(requests_per_minute=int(requests_per_minute))

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    st.caption("Live signal feed")
    feed_placeholder = st.empty()
    feed_lines: list[str] = []

    def on_progress(i: int, total: int, ticker: str) -> None:
        progress_bar.progress(i / total)
        status_text.text(f"[{i}/{total}] {ticker}")

    def on_result(row) -> None:
        feed_lines.insert(
            0, signal_feed_line(row.ticker, row.strategy_name, row.total_return, row.sharpe_ratio, row.error)
        )
        del feed_lines[30:]
        feed_placeholder.markdown(
            f'<div class="signal-feed">{"".join(feed_lines)}</div>', unsafe_allow_html=True
        )

    if walkforward_enabled:
        def on_fold_progress(fold_i: int, fold_total: int, fold_from: str, fold_to: str) -> None:
            progress_bar.progress(fold_i / fold_total)
            status_text.text(f"Fold {fold_i}/{fold_total}: {fold_from} to {fold_to}")

        try:
            fold_results = run_walkforward_scan(
                tickers=tickers,
                strategy_names=strategy_names,
                from_date=str(from_date),
                to_date=str(to_date),
                num_folds=num_folds,
                client=client,
                multiplier=int(multiplier),
                timespan=timespan,
                starting_cash=starting_cash,
                commission_per_trade=commission,
                slippage_bps=slippage_bps,
                max_workers=int(max_workers),
                checkpoint_dir=results_dir / "walkforward",
                fold_progress_callback=on_fold_progress,
                result_callback=on_result,
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Walk-forward scan failed: {e}")
            return

        status_text.text(f"Done — {num_folds} folds, {len(tickers)} tickers.")

        consistency = aggregate_walkforward(fold_results)
        st.subheader("Walk-forward consistency (across folds)")
        st.caption(
            "A strategy with a high mean Sharpe but also high std-across-folds is inconsistent — "
            "it may have gotten lucky in one period rather than having a real edge."
        )
        st.dataframe(pd.DataFrame([asdict(c) for c in consistency]), use_container_width=True)

        with st.expander("Per-fold detail"):
            for fold_label, rows in fold_results.items():
                st.write(f"**{fold_label}**")
                st.dataframe(
                    pd.DataFrame([asdict(a) for a in aggregate_by_strategy(rows)]), use_container_width=True
                )
        return

    try:
        results = run_scan(
            tickers=tickers,
            strategy_names=strategy_names,
            from_date=str(from_date),
            to_date=str(to_date),
            client=client,
            multiplier=int(multiplier),
            timespan=timespan,
            starting_cash=starting_cash,
            commission_per_trade=commission,
            slippage_bps=slippage_bps,
            max_workers=int(max_workers),
            checkpoint_path=checkpoint_path,
            progress_callback=on_progress,
            result_callback=on_result,
            vol_target_enabled=vol_target_enabled,
            target_vol_ann=target_vol_ann,
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"Scan failed: {e}")
        return

    status_text.text(f"Done — {len(tickers)} tickers, {len(results)} results.")

    meta = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe_name,
        "num_tickers": len(tickers),
        "from_date": str(from_date),
        "to_date": str(to_date),
        "multiplier": int(multiplier),
        "timespan": timespan,
        "strategy_names": strategy_names,
        "vol_target_enabled": vol_target_enabled,
        "target_vol_ann": target_vol_ann,
    }
    paths = save_report(results, meta, results_dir)
    record_scan(meta, results)
    st.success("Results saved to Scan History — see the **Scan History** tab. A standalone report is also available below if you want one.")

    aggregates = aggregate_by_strategy(results)
    st.subheader("Per-strategy summary")
    st.dataframe(pd.DataFrame([asdict(a) for a in aggregates]), use_container_width=True)

    ranked = rank_combos(results)
    st.subheader("Top combos")
    st.dataframe(ranked.head(30), use_container_width=True)

    if not ranked.empty:
        top = ranked.iloc[0]
        st.subheader("Backtested equity curve — top-ranked result")
        st.caption(
            f"Historical simulation for {top['ticker']} / {top['strategy_name']} over the window just "
            f"scanned ({from_date} to {to_date}). This is what already happened in the backtest — "
            f"not a forecast or promise of future performance."
        )
        try:
            top_bars = client.get_aggregates(
                ticker=top["ticker"],
                from_date=str(from_date),
                to_date=str(to_date),
                multiplier=int(multiplier),
                timespan=timespan,
            )
            top_strategy = build_strategy(top["strategy_name"])
            top_engine = BacktestEngine(
                starting_cash=starting_cash, commission_per_trade=commission, slippage_bps=slippage_bps
            )
            top_result = top_engine.run(top_bars, top_strategy)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=top_result.equity_curve.index,
                    y=top_result.equity_curve.values,
                    mode="lines",
                    name=f"{top['ticker']} / {top['strategy_name']}",
                )
            )
            fig.update_layout(
                title="Backtested equity curve (top-ranked result)",
                xaxis_title="Time",
                yaxis_title="Equity ($)",
                height=380,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"Could not rebuild the equity curve for the top result: {e}")

    report_text = paths["report"].read_text(encoding="utf-8")
    st.download_button("Download bot_memory.txt", report_text, file_name="bot_memory.txt", mime="text/plain")


@st.cache_data(ttl=60, show_spinner=False)
def _market_status(account_id: str, broker: str) -> dict:
    """Market open/closed for one account, cached 60s so switching tabs / reruns
    don't hammer the broker's clock API. Crypto brokers (no market hours) short-
    circuit to 'always open' without building a client or making any API call."""
    if not BROKER_META.get(broker, {}).get("has_market_hours", True):
        return {"state": "always_open"}
    try:
        account = build_broker_accounts([account_id])[0]
        clock = account.get_market_clock()
    except Exception as e:  # noqa: BLE001
        return {"state": "unknown", "error": str(e)}
    if clock is None:
        return {"state": "always_open"}
    return {"state": "open" if clock["is_open"] else "closed", "next_open": clock.get("next_open")}


@st.cache_data(ttl=30)
def _ibkr_gateway_reachable(host: str, port: int) -> bool:
    """Cached IBKR gateway reachability probe for the Accounts status chip. Kept
    separate from _market_status because get_market_clock() deliberately falls
    back to a heuristic when the gateway is down (so the auto-trader always has a
    clock) — meaning it can't tell us the gateway is actually unreachable."""
    return check_gateway_reachable(host, port)


def _market_status_label(status: dict) -> str:
    state = status["state"]
    if state == "open":
        return "🟢 Market open"
    if state == "always_open":
        return "🟢 Open 24/7"
    if state == "closed":
        nxt = status.get("next_open")
        when = f" · opens {nxt:%a %H:%M} ET" if nxt is not None else ""
        return f"🔴 Market closed{when}"
    return "⚪ Status unavailable"


def _friendly_account_error(exc: Exception) -> str:
    """Plain-English reason for a broker data-fetch failure, so the empty crypto
    placeholder accounts degrade to a neutral note instead of a scary traceback."""
    msg = str(exc)
    if isinstance(exc, NotImplementedError):
        return "not supported by this broker's API"
    if "permission denied" in msg.lower():
        return "the API key lacks the required permission (e.g. Kraken 'Query Funds')"
    if isinstance(exc, AttributeError):
        return "unavailable — the account may be empty or the broker SDK returned an unexpected response"
    return msg[:140]


def _render_live_positions(account_ids: list[str]) -> None:
    """Colour-coded live P&L per open position. st.metric's delta auto-colours
    green (profit) / red (loss) with an up/down arrow and flips as the price
    moves. Called inside an st.fragment so it can auto-refresh on its own."""
    try:
        broker_accounts = build_broker_accounts(account_ids)
    except Exception as e:  # noqa: BLE001
        st.error(f"Failed to connect to linked accounts: {e}")
        return

    st.caption(f"Updated {datetime.now():%H:%M:%S}")
    for broker_account in broker_accounts:
        mode = "Paper" if broker_account.is_paper else "LIVE"
        with st.container(border=True):
            st.write(f"**{broker_account.nickname}** ({mode})")
            try:
                snapshot = broker_account.get_account_snapshot()
                st.caption(
                    f"Equity ${snapshot.equity:,.2f}  |  Cash ${snapshot.cash:,.2f}  |  "
                    f"Buying power ${snapshot.buying_power:,.2f}"
                )
                positions = broker_account.get_positions()
            except Exception as e:  # noqa: BLE001
                st.info(f"Positions {_friendly_account_error(e)}")
                continue

            if not positions:
                st.caption("Flat — no open positions.")
                continue

            cols = st.columns(min(len(positions), 4))
            for i, p in enumerate(positions):
                cost_basis = p.qty * p.avg_entry_price
                pct = (p.unrealized_pl / cost_basis * 100) if cost_basis else 0.0
                # Delta must START with '-' for a loss — Streamlit colours red/green
                # purely on str(delta).startswith("-"), so "$-0.60" would wrongly show
                # green. Put the sign first: "-$0.60" / "+$1.20".
                sign = "-" if p.unrealized_pl < 0 else "+"
                cols[i % 4].metric(
                    f"{p.ticker} · {p.qty:g} sh",
                    f"${p.market_value:,.2f}",
                    delta=f"{sign}${abs(p.unrealized_pl):,.2f} ({pct:+.2f}%)",
                )


def render_accounts_tab() -> None:
    st.subheader("Linked brokerage accounts")
    st.caption(
        "Credentials are encrypted at rest via your OS credential store (Windows Credential "
        "Manager), not stored in plain text — broker_accounts.json (gitignored) only holds "
        "non-secret metadata. New accounts default to paper trading; marking one live "
        "requires an explicit confirmation."
    )

    linked = list_accounts()
    if linked:
        hdr = st.columns([3, 2, 2, 3, 1])
        hdr[0].caption("Account")
        hdr[1].caption("Broker")
        hdr[2].caption("Mode")
        hdr[3].caption("Market")
        for acct in linked:
            cols = st.columns([3, 2, 2, 3, 1])
            cols[0].write(f"**{acct['nickname']}**")
            cols[1].write(BROKER_META.get(acct["broker"], {}).get("label", acct["broker"]))
            cols[2].write("🟢 Paper" if acct["is_paper"] else "🔴 LIVE")
            if acct["broker"] == "ibkr":
                cp = acct.get("conn_params", {})
                if _ibkr_gateway_reachable(cp.get("host", "127.0.0.1"), int(cp.get("port", 4002))):
                    cols[3].write("🟢 Gateway · " + _market_status_label(_market_status(acct["id"], acct["broker"])))
                else:
                    cols[3].write("🔴 Gateway unreachable")
            else:
                cols[3].write(_market_status_label(_market_status(acct["id"], acct["broker"])))
            if cols[4].button("Remove", key=f"remove_{acct['id']}"):
                remove_account(acct["id"])
                st.rerun()
        st.caption(
            "Market status is cached for 60s. Alpaca/IBKR follow the US regular session; crypto trades "
            "24/7. IBKR also shows whether its local gateway is reachable (cached 30s)."
        )
    else:
        st.info("No accounts linked yet.")

    if linked:
        st.divider()
        st.subheader("Account balances")
        bcol1, bcol2 = st.columns([1, 2])
        with bcol1:
            if st.button("Refresh balances"):
                st.session_state["show_balances"] = True
        with bcol2:
            history_period = st.selectbox(
                "History range", ["1W", "1M", "3M", "1Y"], index=1, key="balance_history_period"
            )

        if st.session_state.get("show_balances"):
            try:
                balance_accounts = build_broker_accounts([a["id"] for a in linked])
            except Exception as e:  # noqa: BLE001
                st.error(f"Failed to connect to linked accounts: {e}")
                balance_accounts = []

            balance_rows = []
            history_fig = go.Figure()
            any_history = False
            for broker_account in balance_accounts:
                try:
                    snapshot = broker_account.get_account_snapshot()
                    balance_rows.append(
                        {
                            "account": broker_account.nickname,
                            "mode": "Paper" if broker_account.is_paper else "LIVE",
                            "equity": snapshot.equity,
                            "cash": snapshot.cash,
                            "buying_power": snapshot.buying_power,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    st.info(f"{broker_account.nickname}: balance {_friendly_account_error(e)}")
                    continue

                try:
                    points = broker_account.get_equity_history(period=history_period, timeframe="1D")
                    if points:
                        history_fig.add_trace(
                            go.Scatter(
                                x=[p.timestamp for p in points],
                                y=[p.equity for p in points],
                                mode="lines",
                                name=broker_account.nickname,
                            )
                        )
                        any_history = True
                except Exception as e:  # noqa: BLE001
                    st.caption(f"{broker_account.nickname}: equity history {_friendly_account_error(e)}")

            if balance_rows:
                st.dataframe(pd.DataFrame(balance_rows), use_container_width=True)

            if any_history:
                history_fig.update_layout(
                    title="Account equity over time (all linked accounts)",
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    height=380,
                )
                st.plotly_chart(history_fig, use_container_width=True)
            elif balance_rows:
                st.caption("No balance history available yet for the selected range.")

    if linked:
        st.caption("Live positions with colour-coded P&L are now on the **Overview** page.")

    st.divider()
    st.subheader("Link a new account")
    broker = st.selectbox(
        "Broker", SUPPORTED_BROKERS, format_func=lambda b: BROKER_META[b]["label"], key="link_broker"
    )
    broker_meta = BROKER_META[broker]
    uses_gateway = broker_meta.get("uses_gateway", False)
    cred_label_1, cred_label_2 = broker_meta["cred_fields"]

    with st.form("add_account_form", clear_on_submit=True):
        nickname = st.text_input("Nickname (e.g. 'My Alpaca IRA')")

        if broker_meta["supports_paper"]:
            mode = st.radio("Mode", ["Paper (simulated, recommended)", "Live (real money)"], index=0)
        else:
            st.caption(
                f"⚠ {broker_meta['label']} has no verified sandbox/paper mode in this app — "
                "this account will always connect to the real live API."
            )
            mode = "Live (real money)"

        api_key = secret_key = ""
        ibkr_host = ibkr_account = ""
        ibkr_port, ibkr_client_id = 4002, 1
        if uses_gateway:
            # IBKR connects to a local gateway the user runs and logs into — no API
            # key/secret, just connection config. Rendered as normal (non-masked) inputs.
            st.caption(
                "Interactive Brokers connects to a local **IB Gateway** (or Trader Workstation) that "
                "you run and log into — there's no API key to enter here. Start the gateway, enable its "
                "API (Configure → Settings → API → Enable ActiveX and Socket Clients), and point this at "
                "its host/port. Ports: IB Gateway **4002 paper / 4001 live** (TWS 7497 / 7496)."
            )
            ibkr_host = st.text_input("Gateway host", value="127.0.0.1")
            gwcol1, gwcol2 = st.columns(2)
            with gwcol1:
                ibkr_port = st.number_input(
                    "Gateway port", min_value=1, max_value=65535,
                    value=4002 if mode.startswith("Paper") else 4001, step=1,
                )
            with gwcol2:
                ibkr_client_id = st.number_input("Client ID", min_value=0, value=1, step=1)
            ibkr_account = st.text_input(
                "IBKR account code", placeholder="e.g. DU1234567 (paper) or U1234567 (live)"
            )
        else:
            api_key = st.text_input(cred_label_1, type="password")
            secret_key = st.text_input(cred_label_2, type="password")

        live_confirm = st.checkbox(
            "I understand this connects a REAL brokerage account and orders placed against "
            "it will use real money.",
            disabled=mode.startswith("Paper"),
        )
        submitted = st.form_submit_button("Link account")
        if submitted:
            is_paper = mode.startswith("Paper")
            if uses_gateway:
                if not nickname or not ibkr_account.strip():
                    st.error("Nickname and IBKR account code are required.")
                elif not is_paper and not live_confirm:
                    st.error("Check the confirmation box to link this account.")
                else:
                    add_account(
                        nickname, broker, is_paper,
                        conn_params={
                            "host": ibkr_host.strip() or "127.0.0.1",
                            "port": int(ibkr_port),
                            "client_id": int(ibkr_client_id),
                            "ibkr_account": ibkr_account.strip(),
                        },
                    )
                    st.success(f"Linked {nickname}.")
                    st.rerun()
            else:
                if not nickname or not api_key or not secret_key:
                    st.error(f"Nickname, {cred_label_1}, and {cred_label_2} are all required.")
                elif not is_paper and not live_confirm:
                    st.error("Check the confirmation box to link this account.")
                else:
                    add_account(nickname, broker, is_paper, api_key, secret_key)
                    st.success(f"Linked {nickname}.")
                    st.rerun()


def render_execution_tab() -> None:
    st.subheader("Manual trade execution")
    st.warning(
        "Orders submitted here go straight to the selected brokerage accounts. Nothing in "
        "the Backtest or Scanner tabs triggers this automatically — this is the only place "
        "in the dashboard that places real trades."
    )

    linked = list_accounts()
    if not linked:
        st.info("Link at least one account in the **Accounts** tab first.")
        return

    ticker = st.text_input("Ticker", value="AAPL", key="exec_ticker").upper().strip()
    side = st.radio("Side", ["Buy", "Sell"], horizontal=True, key="exec_side")

    def _apply_size_preset(pct: float) -> None:
        st.session_state["exec_sizing_mode"] = "% of account equity"
        st.session_state["exec_sizing_value"] = pct

    st.caption("Quick size (% of account equity)")
    preset_cols = st.columns(5)
    for col, pct in zip(preset_cols, [5, 10, 25, 50, 100]):
        with col:
            st.button(
                f"{pct}%", key=f"exec_preset_{pct}", width="stretch",
                on_click=_apply_size_preset, args=(float(pct),),
            )

    sizing_label = st.radio(
        "Sizing",
        ["Fixed shares", "% of account equity", "Fixed dollar amount"],
        horizontal=True,
        key="exec_sizing_mode",
    )
    sizing_mode = {
        "Fixed shares": SizingMode.FIXED_SHARES,
        "% of account equity": SizingMode.PCT_EQUITY,
        "Fixed dollar amount": SizingMode.FIXED_DOLLARS,
    }[sizing_label]

    reference_price = None
    if sizing_mode is SizingMode.FIXED_SHARES:
        sizing_value = st.number_input("Quantity (shares)", min_value=0.0001, value=1.0, step=1.0, key="exec_qty")
    else:
        sizing_value = st.number_input(
            "% of equity per account" if sizing_mode is SizingMode.PCT_EQUITY else "Dollar amount per account",
            min_value=0.01,
            value=1.0 if sizing_mode is SizingMode.PCT_EQUITY else 100.0,
            step=0.5 if sizing_mode is SizingMode.PCT_EQUITY else 50.0,
            key="exec_sizing_value",
        )
        reference_price = st.number_input(
            "Reference price ($) — used only to convert this into a share count, "
            "not used for execution itself (the order still fills at market price)",
            min_value=0.01,
            value=100.0,
            step=1.0,
            key="exec_ref_price",
        )

    with st.expander("Optional: bracket order (take-profit / stop-loss)"):
        use_bracket = st.checkbox("Attach take-profit and/or stop-loss", key="exec_use_bracket")
        take_profit_price = None
        stop_loss_price = None
        if use_bracket:
            st.caption(
                "These are absolute $ prices, so set them around the CURRENT market "
                "price — for a buy, take-profit must be above it and stop-loss below "
                "it, or the broker rejects the whole order."
            )
            tp_default = round(reference_price * 1.05, 2) if reference_price else 110.0
            sl_default = round(reference_price * 0.95, 2) if reference_price else 90.0
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                tp_enabled = st.checkbox("Take-profit", key="exec_tp_enabled")
                if tp_enabled:
                    take_profit_price = st.number_input(
                        "Take-profit limit price ($)", min_value=0.01, value=tp_default, key="exec_tp_price"
                    )
            with bcol2:
                sl_enabled = st.checkbox("Stop-loss", key="exec_sl_enabled")
                if sl_enabled:
                    stop_loss_price = st.number_input(
                        "Stop-loss stop price ($)", min_value=0.01, value=sl_default, key="exec_sl_price"
                    )

    account_options = {f"{a['nickname']} ({'Paper' if a['is_paper'] else 'LIVE'})": a["id"] for a in linked}
    selected_labels = st.multiselect("Target accounts", options=list(account_options.keys()), key="exec_accounts")
    selected_ids = [account_options[label] for label in selected_labels]

    if not selected_ids:
        st.info("Select at least one target account.")
        return

    for a in linked:
        if a["id"] in selected_ids:
            status = _market_status(a["id"], a["broker"])
            if status["state"] == "closed":
                st.caption(f"{a['nickname']}: {_market_status_label(status)} — a market order will queue and fill at the next open.")
            else:
                st.caption(f"{a['nickname']}: {_market_status_label(status)}")

    live_selected = [a for a in linked if a["id"] in selected_ids and not a["is_paper"]]

    if st.button("Preview order", key="exec_preview_btn"):
        try:
            broker_accounts = build_broker_accounts(selected_ids)
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to connect to selected accounts: {e}")
            broker_accounts = []

        preview_lines = []
        account_orders: list[AccountOrder] = []
        for broker_account in broker_accounts:
            try:
                qty = compute_qty_for_account(broker_account, reference_price or 0.0, sizing_mode, sizing_value)
            except Exception as e:  # noqa: BLE001
                preview_lines.append(f"  - {broker_account.nickname}: FAILED TO SIZE ({e})")
                continue
            account_orders.append(
                AccountOrder(
                    account=broker_account,
                    qty=qty,
                    take_profit_price=take_profit_price,
                    stop_loss_price=stop_loss_price,
                )
            )
            bracket_note = ""
            if take_profit_price or stop_loss_price:
                bracket_note = f"  [TP {take_profit_price or '-'} / SL {stop_loss_price or '-'}]"
            preview_lines.append(f"  - {broker_account.nickname}: {qty} shares{bracket_note}")

        st.session_state["exec_account_orders"] = account_orders
        st.write("**Order preview:**")
        st.code(f"{side.upper()} {ticker} across {len(account_orders)} account(s):\n" + "\n".join(preview_lines))

    account_orders: list[AccountOrder] = st.session_state.get("exec_account_orders", [])

    live_ack = True
    if live_selected:
        st.error(f"{len(live_selected)} of the selected accounts are LIVE — this will use real money.")
        live_ack = st.checkbox(
            "I understand this will submit real orders with real money on live accounts.",
            key="exec_live_ack",
        )

    if st.button("Submit order", type="primary", disabled=not live_ack or not account_orders):
        order_side = OrderSide.BUY if side == "Buy" else OrderSide.SELL
        with st.spinner("Submitting order to all selected accounts..."):
            results = execute_order_across_accounts(account_orders, ticker, order_side)
        execution_log.log_results(ticker, order_side, results)
        st.session_state["exec_account_orders"] = []

        for r in results:
            if r.success:
                st.success(f"{r.account_nickname}: order accepted by the broker (order id {r.broker_order_id}).")
            else:
                st.error(f"{r.account_nickname}: order REJECTED — {_friendly_broker_error(r.error)}")

        if any(r.success for r in results):
            # account_orders still holds the pre-submit list (session copy was
            # cleared above, this local reference wasn't) — use it for the clock.
            try:
                clock = account_orders[0].account.get_market_clock()
            except Exception:  # noqa: BLE001
                clock = None
            if clock is not None and not clock["is_open"]:
                next_open = clock["next_open"]
                st.info(
                    f"Market is currently CLOSED — accepted orders are queued at the broker and "
                    f"will execute at the next open ({next_open:%A %Y-%m-%d %H:%M} ET). Until then "
                    f"they appear under Alpaca's 'Orders' (not 'Positions')."
                )

        result_rows = [
            {
                "account": r.account_nickname,
                "success": r.success,
                "order_id": r.broker_order_id,
                "filled_qty": r.filled_qty,
                "filled_avg_price": r.filled_avg_price,
                "error": r.error,
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(result_rows), use_container_width=True)

    st.divider()
    st.subheader("Open positions")
    st.caption(
        "One-click close for the selected target accounts: sells the exact held "
        "quantity as a market order, so you never have to retype fractional share counts."
    )
    if st.button("Refresh positions", key="exec_refresh_positions"):
        rows = []
        try:
            for broker_account in build_broker_accounts(selected_ids):
                try:
                    for p in broker_account.get_positions():
                        if p.qty > 0:
                            rows.append(
                                {
                                    "account_id": broker_account.account_id,
                                    "nickname": broker_account.nickname,
                                    "is_paper": broker_account.is_paper,
                                    "ticker": p.ticker,
                                    "qty": p.qty,
                                    "avg_entry_price": p.avg_entry_price,
                                    "market_value": p.market_value,
                                    "unrealized_pl": p.unrealized_pl,
                                }
                            )
                except Exception as e:  # noqa: BLE001
                    st.error(f"{broker_account.nickname}: positions fetch failed: {e}")
        except Exception as e:  # noqa: BLE001
            st.error(f"Failed to connect to selected accounts: {e}")
        st.session_state["exec_open_positions"] = rows
        st.session_state.pop("exec_pending_close", None)

    open_positions = st.session_state.get("exec_open_positions")
    if open_positions is not None:
        if not open_positions:
            st.caption("No open positions on the selected accounts.")
        for row in open_positions:
            pcol1, pcol2, pcol3, pcol4 = st.columns([3, 2, 2, 1])
            mode = "Paper" if row["is_paper"] else "LIVE"
            pcol1.write(f"**{row['ticker']}** — {row['nickname']} ({mode})")
            pcol2.write(f"{row['qty']} sh @ ${row['avg_entry_price']:,.2f}")
            pnl = row["unrealized_pl"]
            pnl_color = "green" if pnl >= 0 else "red"
            pcol3.markdown(
                f"value ${row['market_value']:,.2f} · P&L :{pnl_color}[{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}]"
            )
            with pcol4:
                st.button(
                    "Close", key=f"exec_close_{row['account_id']}_{row['ticker']}",
                    on_click=lambda r=row: st.session_state.__setitem__("exec_pending_close", r),
                )

    pending_close = st.session_state.get("exec_pending_close")
    if pending_close:
        mode = "Paper" if pending_close["is_paper"] else "LIVE"
        st.warning(
            f"Market-sell {pending_close['qty']} {pending_close['ticker']} on "
            f"{pending_close['nickname']} ({mode})?"
        )
        close_ack = True
        if not pending_close["is_paper"]:
            close_ack = st.checkbox(
                "I understand this sells real shares with real money on a live account.",
                key="exec_close_live_ack",
            )
        ccol1, ccol2 = st.columns(2)
        with ccol1:
            if st.button("Confirm close", type="primary", disabled=not close_ack, key="exec_confirm_close"):
                try:
                    account_obj = build_broker_accounts([pending_close["account_id"]])[0]
                    result = account_obj.submit_market_order(
                        pending_close["ticker"], OrderSide.SELL, pending_close["qty"]
                    )
                    execution_log.log_results(pending_close["ticker"], OrderSide.SELL, [result])
                    if result.success:
                        st.success(
                            f"{pending_close['nickname']}: close order accepted "
                            f"(order id {result.broker_order_id})."
                        )
                        attribution = position_attribution.pop_open(
                            pending_close["account_id"], pending_close["ticker"]
                        )
                        if attribution and result.filled_avg_price:
                            live_trades.record_realized_trade(
                                account_id=pending_close["account_id"],
                                ticker=pending_close["ticker"],
                                strategy_name=attribution["strategy_name"],
                                is_paper=pending_close["is_paper"],
                                entry_time=attribution["opened_at"],
                                entry_price=pending_close["avg_entry_price"],
                                exit_time=datetime.now(timezone.utc).isoformat(),
                                exit_price=result.filled_avg_price,
                                qty=result.filled_qty or pending_close["qty"],
                            )
                        try:
                            clock = account_obj.get_market_clock()
                        except Exception:  # noqa: BLE001
                            clock = None
                        if clock is not None and not clock["is_open"]:
                            st.info(
                                f"Market is currently CLOSED — the close order is queued and will "
                                f"execute at the next open ({clock['next_open']:%A %Y-%m-%d %H:%M} ET)."
                            )
                    else:
                        st.error(
                            f"{pending_close['nickname']}: close REJECTED — "
                            f"{_friendly_broker_error(result.error)}"
                        )
                except Exception as e:  # noqa: BLE001
                    st.error(f"Close failed: {e}")
                st.session_state.pop("exec_pending_close", None)
                st.session_state["exec_open_positions"] = None
        with ccol2:
            if st.button("Cancel", key="exec_cancel_close"):
                st.session_state.pop("exec_pending_close", None)
                st.rerun()

    st.divider()
    with st.expander("Execution history (persistent log)"):
        log_df = execution_log.load_log()
        if log_df.empty:
            st.caption("No orders logged yet.")
        else:
            st.dataframe(log_df, use_container_width=True)


def _launch_auto_trader_process() -> None:
    kwargs = {"cwd": str(PROJECT_ROOT)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, str(AUTO_TRADER_SCRIPT)], **kwargs)
    st.session_state["auto_trader_proc"] = proc


def _render_adaptive_roster_section() -> None:
    st.info(
        "Adaptive roster: promotes (ticker, strategy) combos with the best backtest scores, "
        "then automatically pauses any of them that are actually losing money in real "
        "paper/live trading — regardless of what their backtest said. Promotion only happens "
        "when you click **Re-evaluate roster now** below (scanning fresh data is expensive); "
        "pausing on a bad live track record happens automatically, every poll cycle, even "
        "with this dashboard closed."
    )
    st.warning(
        "⚠ Manually trading (via the Trade Execution tab) a ticker currently held by the "
        "active roster on the same account will confuse trade attribution — avoid it while "
        "a ticker is active here."
    )

    state = roster.load_roster()

    st.markdown("#### Roster settings")
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        roster_size = st.number_input(
            "Roster size (max active combos)", min_value=1, value=state.config.roster_size, key="roster_size"
        )
        min_live_trades = st.number_input(
            "Min live trades before live performance counts", min_value=1,
            value=state.config.min_live_trades, key="roster_min_trades",
        )
    with rc2:
        losing_streak_threshold = st.number_input(
            "Pause after N consecutive live losses", min_value=1,
            value=state.config.losing_streak_threshold, key="roster_losing_streak",
        )
        win_rate_floor = st.number_input(
            "Pause if live win rate below (%)", min_value=0.0, max_value=100.0,
            value=state.config.win_rate_floor * 100, step=1.0, key="roster_win_rate_floor",
        )
    with rc3:
        cum_pnl_floor = st.number_input(
            "Pause if live cumulative P&L below ($)", value=state.config.cum_pnl_floor,
            step=50.0, key="roster_cum_pnl_floor",
        )
        max_pnl_drawdown_floor = st.number_input(
            "Pause if live drawdown exceeds ($)", min_value=0.0,
            value=state.config.max_pnl_drawdown_floor, step=50.0, key="roster_max_dd_floor",
        )

    new_config = roster.RosterConfig(
        roster_size=int(roster_size),
        min_live_trades=int(min_live_trades),
        losing_streak_threshold=int(losing_streak_threshold),
        win_rate_floor=win_rate_floor / 100,
        cum_pnl_floor=cum_pnl_floor,
        max_pnl_drawdown_floor=max_pnl_drawdown_floor,
        weights=state.config.weights,
    )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("Save roster settings"):
            saved_state = roster.apply_demotion_checks(state, live_trades.recent_performance, new_config)
            roster.save_roster(saved_state)
            st.success("Roster settings saved (existing entries re-checked against the new thresholds).")
            st.rerun()
    with bcol2:
        if st.button("🔄 Re-evaluate roster now", type="primary"):
            run_id = query_latest_run_id()
            if run_id is None:
                st.error("No scans recorded yet — run one from the **Strategy Scanner** tab first.")
            else:
                scan_rows = query_run_results(run_id)
                new_state = roster.evaluate_roster(scan_rows, live_trades.recent_performance, state, new_config)
                roster.save_roster(new_state)
                st.success(f"Roster re-evaluated against scan run #{run_id} ({len(scan_rows)} results).")
                st.rerun()

    st.markdown("#### Current roster")
    if not state.entries:
        st.caption("Empty — click **Re-evaluate roster now** to populate it from your most recent scan.")
    else:
        rows = []
        for e in state.entries:
            live = e.live_stats or {}
            rows.append(
                {
                    "ticker": e.ticker,
                    "strategy": e.strategy_name,
                    "status": e.status,
                    "backtest_score": round(e.backtest_score, 3),
                    "live_trades": live.get("num_trades"),
                    "live_win_rate": f"{live['win_rate']:.0%}" if live.get("win_rate") is not None else "—",
                    "live_total_pnl": live.get("total_pnl"),
                    "losing_streak": live.get("current_losing_streak"),
                    "reason": e.pause_reason or "",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.markdown("#### Promotion / demotion history")
    events = roster.load_events(limit=50)
    if not events:
        st.caption("No promotions or demotions recorded yet.")
    else:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)

    st.divider()


def render_auto_trading_tab() -> None:
    st.subheader("Automated trading")
    st.warning(
        "This mode executes trades on its own, on a timer, without you reviewing each one. "
        "It runs as a separate process (`auto_trader.py`) coordinated through shared control/"
        "status files, so it keeps running even if you close this dashboard tab. Use the kill "
        "switch below any time to stop it — it's checked before every single trade."
    )

    status = load_status()
    control = load_control()

    st.markdown("### Status")
    actually_running = _render_bot_status(status, control, kill_key="kill_auto")

    st.divider()
    st.markdown("### Configuration")

    linked = list_accounts()
    if not linked:
        st.info("Link at least one account in the **Accounts** tab before configuring auto-trading.")
        return

    mode_label = st.radio(
        "Mode", ["Manual", "Adaptive roster"],
        index=1 if control.use_roster else 0,
        horizontal=True, key="auto_mode",
        help="Manual: one fixed strategy across a fixed ticker list, exactly as before. "
             "Adaptive roster: automatically promotes/demotes (ticker, strategy) combos based "
             "on backtest performance and real live/paper trade outcomes — see below.",
    )
    use_roster = mode_label == "Adaptive roster"

    if not use_roster:
        tickers_input = st.text_input(
            "Tickers to watch (comma-separated — keep this short, each one is polled every cycle "
            "against your Polygon rate limit)",
            value=", ".join(control.tickers) if control.tickers else "AAPL",
            key="auto_tickers",
        )
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

        strategy_name = st.selectbox(
            "Strategy",
            options=list(STRATEGY_REGISTRY.keys()),
            index=list(STRATEGY_REGISTRY.keys()).index(control.strategy_name)
            if control.strategy_name in STRATEGY_REGISTRY
            else 0,
            key="auto_strategy",
        )
    else:
        # Preserve whatever was last saved for manual mode so switching back doesn't lose it —
        # these fields are ignored by auto_trader.py entirely while use_roster=True.
        tickers = control.tickers
        strategy_name = control.strategy_name
        _render_adaptive_roster_section()

    with st.expander("Advanced settings", expanded=False):
        gcol1, gcol2, gcol3 = st.columns(3)
        with gcol1:
            multiplier = st.number_input("Bar multiplier", min_value=1, value=control.multiplier, key="auto_mult")
        with gcol2:
            timespan = st.selectbox(
                "Bar unit", ["minute", "hour", "day"],
                index=["minute", "hour", "day"].index(control.timespan) if control.timespan in ("minute", "hour", "day") else 0,
                key="auto_timespan",
            )
        with gcol3:
            poll_interval = st.number_input(
                "Poll interval (seconds)", min_value=5, value=control.poll_interval_seconds, key="auto_poll"
            )

        sizing_options = ["Fixed shares", "% of account equity", "Fixed dollar amount"]
        sizing_mode_to_label = {
            SizingMode.FIXED_SHARES.value: "Fixed shares",
            SizingMode.PCT_EQUITY.value: "% of account equity",
            SizingMode.FIXED_DOLLARS.value: "Fixed dollar amount",
        }
        sizing_label = st.radio(
            "Sizing", sizing_options,
            index=sizing_options.index(sizing_mode_to_label.get(control.sizing_mode, "Fixed shares")),
            horizontal=True, key="auto_sizing_mode",
        )
        sizing_mode = {
            "Fixed shares": SizingMode.FIXED_SHARES,
            "% of account equity": SizingMode.PCT_EQUITY,
            "Fixed dollar amount": SizingMode.FIXED_DOLLARS,
        }[sizing_label]
        sizing_value = st.number_input(
            "Sizing value (shares, % of equity, or $ depending on the mode above)",
            min_value=0.01, value=control.sizing_value, key="auto_sizing_value",
        )

        max_trades_per_day = st.number_input(
            "Max trades per day (hard limit, resets at UTC midnight)",
            min_value=1, value=control.max_trades_per_day, key="auto_max_trades",
        )

        st.caption("GARCH volatility filter + sizing")
        vol_target_enabled = st.checkbox(
            "Enable: block new BUYs while a ticker is in a high-vol (\"storm\") regime, and scale "
            "the sizing value above by forecast vol otherwise. Recomputed once per ticker per day "
            "(not every poll). Off by default.",
            value=control.vol_target_enabled, key="auto_vol_target",
        )
        vol_target_ann = st.number_input(
            "Target annualized vol (%)", min_value=1.0, value=control.vol_target_ann, step=1.0,
            key="auto_vol_target_ann", disabled=not vol_target_enabled,
        )

    account_options = {f"{a['nickname']} ({'Paper' if a['is_paper'] else 'LIVE'})": a["id"] for a in linked}
    selected_labels = st.multiselect(
        "Target accounts", options=list(account_options.keys()),
        default=[lbl for lbl, aid in account_options.items() if aid in control.account_ids],
        key="auto_accounts",
    )
    selected_ids = [account_options[label] for label in selected_labels]
    live_selected = [a for a in linked if a["id"] in selected_ids and not a["is_paper"]]

    allow_live = control.allow_live
    if live_selected:
        st.error(
            f"{len(live_selected)} selected account(s) are LIVE. Auto-trading a live account with "
            "no human reviewing each trade is the highest-risk mode this dashboard has."
        )
        arm_phrase = st.text_input(
            f'Type exactly "{LIVE_ARM_PHRASE}" to allow auto-trading to target these live accounts',
            key="auto_live_arm_phrase",
        )
        allow_live = arm_phrase.strip() == LIVE_ARM_PHRASE
        if arm_phrase and not allow_live:
            st.warning("Phrase doesn't match — live accounts will be excluded from auto-trading until it does.")
    else:
        allow_live = False

    st.caption(
        "The account-level max-drawdown circuit breaker now lives in **Settings → Account "
        "risk limit** (it applies to both Manual and Adaptive mode)."
    )

    if st.button("Save configuration"):
        new_control = AutoTraderControl(
            enabled=control.enabled,
            killed=control.killed,
            allow_live=allow_live,
            account_ids=selected_ids,
            tickers=tickers,
            strategy_name=strategy_name,
            multiplier=int(multiplier),
            timespan=timespan,
            poll_interval_seconds=int(poll_interval),
            max_trades_per_day=int(max_trades_per_day),
            sizing_mode=sizing_mode.value,
            sizing_value=sizing_value,
            vol_target_enabled=vol_target_enabled,
            vol_target_ann=vol_target_ann,
            use_roster=use_roster,
            # Owned by the Settings page now — carry forward from the loaded control so
            # saving here never clobbers the account-risk limit set in Settings.
            max_drawdown_enabled=control.max_drawdown_enabled,
            max_drawdown_pct=control.max_drawdown_pct,
        )
        save_control(new_control)
        st.success("Configuration saved.")
        st.rerun()

    st.divider()
    st.markdown("### Start / Stop")

    if not selected_ids or (not use_roster and not tickers):
        st.info(
            "Set at least one target account (and, in Manual mode, at least one ticker), "
            "then save the configuration, to start."
        )
        return

    scol1, scol2 = st.columns(2)
    with scol1:
        if st.button("▶ Start auto-trading", type="primary"):
            armed_control = load_control()
            armed_control.enabled = True
            armed_control.killed = False
            save_control(armed_control)
            if not actually_running:
                _launch_auto_trader_process()
            st.success("Auto-trading armed. It will start trading on its next poll cycle.")
            st.rerun()
    with scol2:
        if st.button("⏸ Pause (disable, keep process alive)"):
            paused_control = load_control()
            paused_control.enabled = False
            save_control(paused_control)
            st.success("Paused — the process stays running but won't submit new trades until re-armed.")
            st.rerun()


def render_scan_history_tab() -> None:
    st.subheader("Everything the scanner has ever found")
    st.caption(
        "Every Strategy Scanner run gets added here automatically — nothing to download, "
        "nothing to lose between sessions. This accumulates across every run, not just the last one."
    )

    counts = query_summary_counts()
    if counts["num_runs"] == 0:
        st.info("No scans recorded yet — run one from the **Strategy Scanner** tab and it'll show up here.")
        return

    with st.container(horizontal=True):
        st.metric("Scan runs", counts["num_runs"], border=True)
        st.metric("Results recorded", counts["num_results"], border=True)
        st.metric("Distinct tickers", counts["num_tickers"], border=True)
        st.metric("Strategies tested", counts["num_strategies"], border=True)

    st.divider()
    st.markdown("### Strategy leaderboard")
    st.caption("Aggregated across every run ever recorded — not just the most recent scan.")
    leaderboard = query_strategy_leaderboard()
    st.dataframe(
        leaderboard,
        width="stretch",
        hide_index=True,
        column_config={
            "strategy_name": st.column_config.TextColumn("Strategy"),
            "num_results": st.column_config.NumberColumn("Results"),
            "num_runs": st.column_config.NumberColumn("Runs"),
            "mean_sharpe": st.column_config.NumberColumn("Mean Sharpe", format="%.2f"),
            "mean_return": st.column_config.NumberColumn("Mean return", format="percent"),
            "pct_profitable": st.column_config.NumberColumn("% profitable", format="percent"),
            "mean_conviction": st.column_config.NumberColumn("Mean conviction", format="%.2f", help="Average entry-signal conviction (0-1) across this strategy's trades. Logged for future learning; does not affect sizing."),
        },
    )

    st.divider()
    st.markdown("### All-time top combos")
    all_results = query_all_time_top(limit=10_000)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        ticker_filter = st.selectbox(
            "Filter by ticker", options=["(all)"] + sorted(all_results["ticker"].unique().tolist()),
            key="history_ticker_filter",
        )
    with fcol2:
        strategy_filter = st.selectbox(
            "Filter by strategy", options=["(all)"] + sorted(all_results["strategy_name"].unique().tolist()),
            key="history_strategy_filter",
        )
    top = query_all_time_top(
        limit=100,
        ticker=None if ticker_filter == "(all)" else ticker_filter,
        strategy_name=None if strategy_filter == "(all)" else strategy_filter,
    )
    st.dataframe(
        top,
        width="stretch",
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "strategy_name": st.column_config.TextColumn("Strategy"),
            "total_return": st.column_config.NumberColumn("Return", format="percent"),
            "cagr": st.column_config.NumberColumn("CAGR", format="percent"),
            "max_drawdown": st.column_config.NumberColumn("Max drawdown", format="percent"),
            "sharpe_ratio": st.column_config.NumberColumn("Sharpe", format="%.2f"),
            "num_trades": st.column_config.NumberColumn("Trades"),
            "win_rate": st.column_config.NumberColumn("Win rate", format="percent"),
            "conviction": st.column_config.NumberColumn("Conviction", format="%.2f", help="Average entry-signal conviction (0-1) for this combo. Logged for future learning; does not affect sizing."),
            "run_at": st.column_config.DatetimeColumn("Scanned at"),
        },
    )

    st.divider()
    st.markdown("### Recent runs")
    recent = query_recent_runs(limit=20)
    st.dataframe(
        recent,
        width="stretch",
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("Run"),
            "run_at": st.column_config.DatetimeColumn("Ran at"),
            "universe": st.column_config.TextColumn("Universe"),
            "num_tickers": st.column_config.NumberColumn("Tickers"),
            "from_date": st.column_config.TextColumn("From"),
            "to_date": st.column_config.TextColumn("To"),
            "multiplier": st.column_config.NumberColumn("Bar mult."),
            "timespan": st.column_config.TextColumn("Bar unit"),
            "strategy_names": None,
        },
    )

    st.divider()
    with st.expander("Clear history"):
        st.caption("Permanently deletes every recorded run and result. This can't be undone.")
        confirm_clear = st.checkbox("I understand this deletes all scan history permanently.", key="history_clear_confirm")
        if st.button("Clear all scan history", disabled=not confirm_clear):
            clear_history()
            st.success("Scan history cleared.")
            st.rerun()


def _render_bot_status(status, control, *, kill_switch: bool = True, kill_key: str = "kill_switch_btn") -> bool:
    """Auto-trader status row (process / armed / trades today / killed) plus the
    last signal/error and an optional kill switch. Shared by the Overview home
    and the Auto Trading tab so both always show the same truth. Returns whether
    the process is actually running (heartbeat-fresh), which the Auto Trading tab
    uses to decide whether to launch a new process on Start."""
    heartbeat_stale = True
    if status.last_heartbeat:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
            heartbeat_stale = age > max(control.poll_interval_seconds * 3, 30)
        except ValueError:
            heartbeat_stale = True
    actually_running = status.running and not heartbeat_stale

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Process", "🟢 Running" if actually_running else "🔴 Not running")
    c2.metric("Armed", "🟢 Yes" if control.enabled and not control.killed else "🔴 No")
    c3.metric("Trades today", f"{status.trades_today} / {control.max_trades_per_day}")
    c4.metric("Killed", "⚠ Yes" if control.killed else "No")

    if status.last_signal:
        st.caption(f"Last signal: {status.last_signal}")
    if status.last_error:
        st.error(f"Last error: {status.last_error}")

    if kill_switch and st.button("🛑 KILL SWITCH — stop auto-trading now", type="primary", key=kill_key):
        trigger_kill_switch()
        st.success("Kill switch engaged. The trader will stop within one poll cycle.")
        st.rerun()

    return actually_running


def render_overview_page() -> None:
    """Daily-glance home: bot status, open positions with live P&L, and recent
    trades on one screen, with the kill switch always reachable. Reuses the same
    helpers as the Auto Trading / Accounts tabs — this page only re-arranges."""
    st.subheader("Overview")
    st.caption("Your daily glance — bot status, open positions, and recent trades on one screen.")

    status = load_status()
    control = load_control()
    _render_bot_status(status, control, kill_key="kill_overview")

    st.divider()
    st.markdown("### Open positions")
    linked = list_accounts()
    target_ids = [a["id"] for a in linked if a["id"] in control.account_ids]
    if not target_ids:
        st.info(
            "No auto-trading accounts are configured yet. Link one under **Accounts**, then set it "
            "as a target under **Auto trading** — its live positions will show here."
        )
    else:
        any_live = any(not a["is_paper"] for a in linked if a["id"] in target_ids)

        @st.fragment(run_every=(15 if any_live else None))
        def _overview_positions() -> None:
            _render_live_positions(target_ids)

        _overview_positions()

    st.divider()
    st.markdown("### Recent trades")
    log_df = execution_log.load_log(limit=10)
    if log_df.empty:
        st.caption(
            "No executed trades logged yet — either the bot hasn't traded, or no manual order "
            "has been placed."
        )
    else:
        st.dataframe(log_df, hide_index=True, width="stretch")


def render_backtest_page() -> None:
    """Backtest tab wrapped as a navigation page: render the form, then run any
    pending backtest inline. The old 'defer to the end of main()' workaround only
    existed to stop sibling tabs looking stuck while a backtest blocked the script
    — under single-page navigation there are no sibling tabs, so it's unnecessary."""
    pending = render_backtest_tab()
    if pending is not None:
        params, results_container = pending
        execute_backtest(params, results_container)


def main() -> None:
    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)
    authenticator, username = require_auth()

    # Grouped sidebar navigation (Overview / Research / Trading / Settings). Each
    # page is an existing render_* function wrapped as a callable st.Page — the
    # reorg re-arranges placement only, it does not rewrite any tab's logic.
    pages = {
        "": [
            st.Page(render_overview_page, title="Overview", icon=":material/dashboard:", default=True),
        ],
        "Research": [
            st.Page(render_backtest_page, title="Backtest", icon=":material/science:"),
            st.Page(render_scanner_tab, title="Strategy scanner", icon=":material/radar:"),
            st.Page(render_scan_history_tab, title="Scan history", icon=":material/history:"),
        ],
        "Trading": [
            st.Page(render_accounts_tab, title="Accounts", icon=":material/account_balance:"),
            st.Page(render_execution_tab, title="Trade execution", icon=":material/sync_alt:"),
            st.Page(render_auto_trading_tab, title="Auto trading", icon=":material/smart_toy:"),
        ],
        "Settings": [
            st.Page(render_settings_page, title="Settings", icon=":material/settings:"),
        ],
    }
    page = st.navigation(pages, position="sidebar")

    with st.sidebar:
        st.divider()
        st.write(f"Logged in as **{username}**")
        authenticator.logout("Log out", "sidebar")

    st.title("Trading Bot Backtester")
    page.run()


if __name__ == "__main__":
    main()
