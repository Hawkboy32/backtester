"""Password-protected Streamlit dashboard for the backtester.

Run with:  streamlit run app.py
First-time setup (creates your login):  python scripts/setup_auth.py
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from streamlit_authenticator.utilities.exceptions import LoginError

from backtester import events, execution_log, keystore, two_factor
from backtester.accounts import (
    BROKER_META,
    SUPPORTED_BROKERS,
    account_asset_class,
    add_account,
    build_broker_accounts,
    infer_asset_class,
    list_accounts,
    remove_account,
)
from backtester.auto_trader_state import AutoTraderControl, load_control, load_status, save_control, trigger_kill_switch
from backtester.brokers.base import OrderSide
from backtester.brokers.ibkr import check_gateway_reachable
from backtester.data import DEFAULT_CACHE_DIR, PolygonClient, PolygonError, cache_stats, prune_cache
from backtester.engine import ENGINE_VERSION, BacktestEngine
from backtester.execution import AccountOrder, SizingMode, compute_qty_for_account, execute_order_across_accounts
from backtester.live_trades import realized_pnl_by_account
from backtester.memory_report import save_report
from backtester.metrics import (
    MARKET_CALENDARS,
    classify_ticker_regime,
    compute_report,
    periods_per_year_for_calendar,
)
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
from backtester.strategies import STRATEGY_REGISTRY, build_strategy, strategy_regime
from backtester.strategies.sma_crossover import SmaCrossoverStrategy
from backtester.universe import UNIVERSE_REGISTRY, load_universe, sample_universe, sector_for_ticker
from backtester.walkforward import aggregate_walkforward, run_walkforward_scan
from backtester import account_risk, app_settings, daily_pnl_guard, heartbeat, live_trades, notifications, playlist, position_attribution, roster, volatility

PROJECT_ROOT = Path(__file__).resolve().parent
AUTO_TRADER_SCRIPT = PROJECT_ROOT / "auto_trader.py"
SCAN_RUNNER_SCRIPT = PROJECT_ROOT / "scan_runner.py"
LIVE_ARM_PHRASE = "I ARM LIVE AUTO-TRADING"

st.set_page_config(page_title="Holotable", layout="wide")

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
    try:
        authenticator.login("main")
    except LoginError:
        # A saved session cookie naming a user who no longer exists in
        # auth_config.yaml (e.g. a temporary account that was since removed)
        # otherwise raises straight out of login() and replaces the whole page
        # with a traceback — no login form, no way back in without knowing to
        # clear cookies by hand. Drop the stale cookie and show the form.
        authenticator.cookie_controller.delete_cookie()
        st.warning("Your saved session is no longer valid. Please log in again.")
        st.stop()

    auth_status = st.session_state.get("authentication_status")
    if auth_status is not True:
        # Not (or no longer) password-authenticated — drop any stale TOTP-passed
        # flag from a previous login in this same browser session, so a second
        # login (e.g. log out, log back in as someone else) always re-challenges
        # 2FA rather than silently trusting the first login's verification.
        st.session_state.pop("totp_verified_user", None)
    if auth_status is False:
        st.error("Username or password is incorrect.")
        st.stop()
    if auth_status is None:
        st.warning("Please log in to continue.")
        st.stop()

    username = st.session_state.get("username", "")

    if two_factor.is_enrolled(username) and st.session_state.get("totp_verified_user") != username:
        _render_totp_challenge(username)
        st.stop()

    return authenticator, username


def _render_totp_challenge(username: str) -> None:
    """Second-factor gate shown after a successful password login, once per
    Streamlit session (not re-asked on every rerun/page click — see the
    totp_verified_user flag in require_auth). Only ever reached for a user who
    has completed enrolment (two_factor.is_enrolled), so this can never lock
    out someone who hasn't set 2FA up."""
    st.title("Holotable")
    st.caption("Trading Bot Backtester")
    st.subheader("Two-factor authentication")
    st.caption("Enter the 6-digit code from your authenticator app.")

    with st.form("totp_challenge_form"):
        code = st.text_input("Authentication code", max_chars=10)
        submitted = st.form_submit_button("Verify")
    if submitted:
        if two_factor.verify_code(code, username=username):
            st.session_state["totp_verified_user"] = username
            st.rerun()
        else:
            st.error("Incorrect code. Please try again.")

    with st.expander("Use a recovery code instead"):
        st.caption(
            f"{two_factor.remaining_recovery_codes(username)} recovery code(s) remaining. "
            "Each one can only be used once."
        )
        with st.form("totp_recovery_form"):
            recovery_code = st.text_input("Recovery code")
            recovery_submitted = st.form_submit_button("Use recovery code")
        if recovery_submitted:
            if two_factor.consume_recovery_code(username, recovery_code):
                st.session_state["totp_verified_user"] = username
                st.warning(
                    f"Recovery code accepted — {two_factor.remaining_recovery_codes(username)} remaining. "
                    "Consider re-enrolling 2FA from Settings soon to get a fresh set."
                )
                st.rerun()
            else:
                st.error("Invalid or already-used recovery code.")


def render_settings_page() -> None:
    """One home for all configuration: API keys, phone notifications, the
    bot-down dead-man's switch, the execution-cost defaults new Backtest/Scanner
    runs start from, and the account-level max-drawdown circuit breaker.
    Consolidated here so settings stop being scattered across the API Keys tab,
    the Backtest/Scanner sidebars, and the Auto Trading tab."""
    st.subheader("Settings")
    st.caption("API keys, notifications, the bot-down alert, execution-cost defaults, and the account-risk limit — all in one place.")

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
        "Get a push to your phone for trade opens/closes, order rejections, and risk-limit "
        "trips (drawdown breaker, daily giveback guard, kill switch). Uses ntfy: install the "
        "**ntfy** app (iOS/Android), then in the app subscribe to the exact topic name you set "
        "below. No account or password needed — so pick a long, unguessable topic name (anyone "
        "who knows it can see your pings), or self-host your own server below for real privacy."
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
    with st.expander("Self-hosted ntfy server (optional)"):
        st.caption(
            "The public ntfy.sh server is unauthenticated pub-sub — anyone who guesses your "
            "topic name can subscribe and see real trade activity/P&L. Point this at your own "
            "self-hosted ntfy instance (e.g. reachable over Tailscale) for real privacy. Leave "
            "as the default if you don't have one set up."
        )
        notif_base = st.text_input(
            "ntfy server URL",
            value=notif_cfg.get("ntfy_base", notifications.DEFAULT_NTFY_BASE),
            key="notif_base",
        )
    ncol1, ncol2 = st.columns(2)
    with ncol1:
        if st.button("Save notification settings", key="notif_save"):
            notifications.save_config(
                {
                    "enabled": notif_enabled, "provider": "ntfy",
                    "ntfy_topic": notif_topic.strip(),
                    "ntfy_base": notif_base.strip() or notifications.DEFAULT_NTFY_BASE,
                }
            )
            st.success("Notification settings saved.")
    with ncol2:
        if st.button("Send test notification", key="notif_test"):
            ok = notifications.notify(
                "Test notification",
                "If you can read this on your phone, trade alerts are working.",
                config={
                    "enabled": True, "provider": "ntfy",
                    "ntfy_topic": notif_topic.strip(),
                    "ntfy_base": notif_base.strip() or notifications.DEFAULT_NTFY_BASE,
                },
            )
            if ok:
                st.success("Sent — check your phone. If nothing arrives, confirm the app is subscribed to that exact topic.")
            elif not notif_topic.strip():
                st.error("Enter a topic name first.")
            else:
                st.error("Send failed — check your internet connection, the topic name, and the server URL.")

    st.divider()
    st.markdown("### Bot-down alert (dead-man's switch)")
    st.caption(
        "Alerts you when the bot **stops** running — the one failure the phone notifications "
        "above can't catch, because a dead bot sends nothing and silence looks just like a quiet "
        "trading day. (On 22 July the bot died when the laptop slept and stayed down ~29 hours "
        "with a position open; nothing alerted.) A watchdog on this laptop can't fix that either "
        "— whatever kills the bot kills the watchdog. So the check is inverted: the bot pings an "
        "outside service every minute, and **that service** alerts you when the pings stop."
    )
    with st.expander("How to set this up (one-time, free)", expanded=False):
        st.markdown(
            """
1. Create a free account at **healthchecks.io** and add a new check.
2. Copy its **ping URL** (looks like `https://hc-ping.com/<uuid>`) and paste it below.
3. On the check's page, set **Schedule → Cron** to the hours the bot is meant to be up, e.g.
   `*/5 13-21 * * 1-5` (UTC — roughly US market hours, weekdays only), with a grace period
   of ~15 minutes.
4. Add your phone/email under **Notification methods** on healthchecks.io.

**Step 3 is the one that matters.** You shut the laptop down overnight, so the bot is
*supposed* to go quiet then. Without a schedule you'd get paged every single night, and an
alert you learn to ignore is worse than no alert at all. With it, silence at 3pm on a
Tuesday pages you and silence at 3am doesn't.

Pausing the check on that site is also how you silence it for a planned break — the bot has
no way to say "I'm off on purpose".
            """
        )
    hb_cfg = heartbeat.load_config()
    hb_enabled = st.checkbox(
        "Enable bot-down alert", value=hb_cfg.get("enabled", False), key="hb_enabled"
    )
    hb_url = st.text_input(
        "Ping URL",
        value=hb_cfg.get("ping_url", ""),
        key="hb_url",
        help=(
            "From healthchecks.io (or any compatible/self-hosted monitor). Kept in a local "
            "gitignored file. It can't trade or move money — worst case someone with the URL "
            "could fake an 'alive' ping — so it isn't stored in the credential manager."
        ),
    )
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        if st.button("Save bot-down alert settings", key="hb_save"):
            if hb_enabled and not heartbeat.is_valid_ping_url(hb_url):
                st.error("That doesn't look like a ping URL — it should start with https://")
            else:
                heartbeat.save_config({"enabled": hb_enabled, "ping_url": hb_url.strip()})
                st.success("Bot-down alert settings saved. Takes effect on the bot's next cycle.")
    with hcol2:
        if st.button("Send test ping", key="hb_test"):
            if not heartbeat.is_valid_ping_url(hb_url):
                st.error("Enter a valid https:// ping URL first.")
            elif heartbeat.ping(
                config={"enabled": True, "ping_url": hb_url.strip()}, force=True
            ):
                st.success(
                    "Ping delivered — the check on healthchecks.io should now show as up. "
                    "Confirm it went green there before relying on it."
                )
            else:
                st.error("Ping failed — check the URL and your internet connection.")

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
    watched_ids = _all_target_account_ids(control)
    watched = [a for a in linked if a["id"] in watched_ids]
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
    if watched_ids and not blocked_any:
        st.caption("No target accounts are currently risk-blocked.")

    st.divider()
    st.markdown("### Daily P&L giveback guard")
    st.caption(
        "A lighter, DAILY counterpart to the breaker above. Once an account's profit for today "
        "has given back this much of today's own peak profit, new entries block for the rest of "
        "the day (existing positions can still be closed) — protects gains on a hot day without "
        "capping its upside, and without needing an outright loss to trigger. Unlike the breaker "
        "above, it resets itself automatically at the start of the next trading day — no re-arm "
        "needed."
    )
    giveback_enabled = st.checkbox(
        "Enable daily P&L giveback guard. Off by default.",
        value=control.giveback_enabled, key="settings_giveback_enabled",
    )
    giveback_pct = st.number_input(
        "Giveback of today's peak profit that blocks new entries (%)", min_value=1.0, max_value=100.0,
        value=control.giveback_pct, step=5.0, key="settings_giveback_pct", disabled=not giveback_enabled,
    )
    if st.button("Save daily P&L giveback guard", key="settings_save_giveback"):
        fresh = load_control()
        fresh.giveback_enabled = giveback_enabled
        fresh.giveback_pct = giveback_pct
        save_control(fresh)
        st.success("Daily P&L giveback guard saved.")
        st.rerun()

    giveback_blocked_any = False
    for account in watched:
        gb_status = daily_pnl_guard.get_status(account["id"])
        if gb_status and gb_status.get("blocked"):
            giveback_blocked_any = True
            st.warning(f"{account['nickname']}: giveback-blocked today — {gb_status['reason']}")
    if watched_ids and not giveback_blocked_any:
        st.caption("No target accounts are currently giveback-blocked today.")

    st.divider()
    st.markdown("### Data cache")
    st.caption(
        "Every unique (ticker, date range, granularity) bar fetch gets pickled to disk so scans "
        "are resumable and don't re-fetch data they already have — but nothing ever removes an "
        "old entry, so it grows forever as scans get re-run over shifting date windows."
    )
    file_count, total_bytes = cache_stats(DEFAULT_CACHE_DIR)
    st.caption(f"Current cache: {file_count} file(s), {total_bytes / 1_048_576:.1f} MB")
    cache_col1, cache_col2 = st.columns([2, 1])
    with cache_col1:
        max_age_days = st.number_input(
            "Delete cache files older than (days)", min_value=1, max_value=365,
            value=30, step=1, key="settings_cache_max_age",
        )
    with cache_col2:
        st.write("")  # vertical alignment with the number_input's label above
        if st.button("Prune old cache files", key="settings_prune_cache"):
            deleted, freed = prune_cache(DEFAULT_CACHE_DIR, max_age_days=int(max_age_days))
            if deleted:
                st.success(f"Deleted {deleted} file(s), freed {freed / 1_048_576:.1f} MB.")
            else:
                st.info("Nothing older than that to delete.")
            st.rerun()

    st.divider()
    _render_2fa_settings(st.session_state.get("username", ""))


def _render_2fa_settings(username: str) -> None:
    """Enrolment UI for src/backtester/two_factor.py — never enforced until
    enrolment completes (a mis-scanned QR can never lock you out), and there's
    deliberately no disable button here: the module's own documented recovery
    path is a local-terminal `two_factor.disable(username)` call, not a
    dashboard toggle — see that module's docstring for why."""
    st.markdown("### Two-factor authentication (2FA)")
    if not username:
        st.caption("Log in to manage 2FA.")
        return

    if two_factor.is_enrolled(username):
        st.success(
            f"2FA is enabled for **{username}**. "
            f"{two_factor.remaining_recovery_codes(username)} recovery code(s) remaining."
        )
        st.caption(
            "To disable or reset 2FA, run this from a terminal on this machine: "
            f'`python -c "from backtester import two_factor; two_factor.disable(\'{username}\')"`'
        )
        return

    st.caption(
        "Adds a 6-digit code (Google Authenticator, Authy, 1Password, etc.) on top of your "
        "password — matters most once this dashboard is reachable from somewhere other than "
        "this machine (e.g. over Tailscale), since a password alone would otherwise be enough "
        "to control the bot."
    )

    if "totp_enrol_secret" not in st.session_state:
        if st.button("Set up 2FA", key="totp_begin_enrol"):
            secret, uri = two_factor.begin_enrolment(username)
            st.session_state["totp_enrol_secret"] = secret
            st.session_state["totp_enrol_uri"] = uri
            st.rerun()
        return

    secret = st.session_state["totp_enrol_secret"]
    uri = st.session_state["totp_enrol_uri"]
    st.write("**1. Scan this QR code** in your authenticator app (or enter the key manually).")
    qr_img = qrcode.make(uri)
    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    st.image(buf.getvalue(), width=220)
    st.caption(f"Manual setup key: `{secret}`")

    st.write("**2. Enter the 6-digit code** your app is now showing, to confirm it's set up correctly.")
    with st.form("totp_complete_enrol_form"):
        confirm_code = st.text_input("Authentication code", max_chars=10, key="totp_confirm_code")
        confirm_submitted = st.form_submit_button("Confirm and enable 2FA")
    if confirm_submitted:
        codes = two_factor.complete_enrolment(username, secret, confirm_code)
        if codes is None:
            st.error("Incorrect code — check your app and try again.")
        else:
            del st.session_state["totp_enrol_secret"]
            del st.session_state["totp_enrol_uri"]
            st.session_state["totp_verified_user"] = username  # already proved possession this session
            st.success("2FA enabled. Save these recovery codes now — this is the only time they're shown:")
            st.code("\n".join(codes))
            st.warning("Each code works once. Store them somewhere safe (password manager, printed copy) — not in this project's repo.")

    if st.button("Cancel setup", key="totp_cancel_enrol"):
        del st.session_state["totp_enrol_secret"]
        del st.session_state["totp_enrol_uri"]
        st.rerun()


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


_ASSET_CLASS_UNIVERSES = {
    "equity": ["S&P 500", "Nasdaq-100 (US Tech 100)"],
    "crypto": ["Crypto (top 15 USD pairs)"],
    "forex": ["Forex (7 major USD pairs)"],
}


def _known_tickers_for_asset_class(asset_class: str) -> dict[str, str]:
    """Same shape as _known_tickers(), scoped to only the universe(s) matching
    one asset class — the Trade Execution tab's compatibility-gated ticker
    picker (see render_execution_tab) uses this instead of the combined list,
    so an equity account can't even be offered a forex/crypto ticker."""
    known: dict[str, str] = {}
    for universe_name in _ASSET_CLASS_UNIVERSES[asset_class]:
        try:
            df = load_universe(universe_name)
        except FileNotFoundError:
            continue
        for _, row in df.iterrows():
            name = row["name"] if "name" in row and isinstance(row["name"], str) else ""
            known.setdefault(row["ticker"], name)
    return known


def _all_target_account_ids(control: AutoTraderControl) -> set[str]:
    """Every account id the auto-trader could touch this cycle: the primary
    control.account_ids PLUS every extra_targets group's own account_ids
    (see auto_trader.py's _resolve_targets/run_cycle, which unions the same
    way). Settings' risk/giveback-blocked lists and Overview's positions
    panel both need this union too, or an extra target's account (e.g. a
    forex IG account) is correctly protected by the engine-level guards but
    invisible and non-re-armable from the dashboard."""
    return set(control.account_ids) | {
        aid for group in control.extra_targets for aid in group.get("account_ids", [])
    }


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

        st.divider()
        event_filter_enabled = st.checkbox(
            "Skip entries on FOMC days — proactive step-out around known risk "
            "events (exits still fire). Off by default so existing results stay "
            "comparable.",
            value=False, key="bt_event_filter",
        )

        run_clicked = st.button("Run backtest", type="primary", width="stretch")

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
        "event_filter_enabled": event_filter_enabled,
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
    event_filter_enabled = params["event_filter_enabled"]
    # Polygon's own ticker namespace, not a guess: crypto tickers are always
    # prefixed "X:" and forex "C:" (vs a bare equity symbol) — see
    # metrics.MARKET_CALENDARS for why they need different annualization
    # (crypto: 24/7, 365 days; forex: 24h session but closed weekends, 252
    # days), same reasoning as the Scanner tab's universe-derived pick.
    if ticker.startswith("X:"):
        market_calendar = "crypto"
    elif ticker.startswith("C:"):
        market_calendar = "forex"
    else:
        market_calendar = "equity"

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
                    regime_table = volatility.compute_regime_table(
                        daily_bars, target_vol_ann=target_vol_ann,
                        periods_per_year=MARKET_CALENDARS[market_calendar][1],
                    )
                    regime_by_date = volatility.regime_by_date(regime_table)
                    latest_regime = volatility.latest_regime_info(regime_table)
                except volatility.InsufficientHistoryError as e:
                    st.warning(f"Not enough daily history for a GARCH regime on {ticker}: {e} Running without it.")
                except PolygonError as e:
                    st.warning(f"Daily history fetch failed for the GARCH regime: {e} Running without it.")

            st.write("Running strategy over bars...")
            blocked_dates = None
            if event_filter_enabled:
                blocked_dates = events.blocked_dates_in_range(
                    pd.Timestamp(from_date).date(), pd.Timestamp(to_date).date()
                )
                if blocked_dates:
                    st.write(f"Event step-out: skipping entries on {len(blocked_dates)} FOMC day(s) in this window.")
            strategy = SmaCrossoverStrategy(fast_window=int(fast_window), slow_window=int(slow_window))
            engine = BacktestEngine(
                starting_cash=starting_cash,
                commission_per_trade=commission,
                slippage_bps=slippage_bps,
                regime_by_date=regime_by_date,
                blocked_dates=blocked_dates,
            )
            result = engine.run(bars, strategy)

            st.write("Computing performance metrics...")
            ppy = periods_per_year_for_calendar(timespan, int(multiplier), market_calendar)
            report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)

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

    if report.expectancy is not None:
        e1, e2, e3 = st.columns(3)
        e1.metric("Expectancy / trade", f"{report.expectancy:+.3%}",
                  help="Win% × avg win + loss% × avg loss — what an average trade returned. Positive = the maths is on your side ('be the casino').")
        e2.metric("Avg winning trade", f"{report.avg_win_pct:+.2%}" if report.avg_win_pct is not None else "—")
        e3.metric("Avg losing trade", f"{report.avg_loss_pct:+.2%}" if report.avg_loss_pct is not None else "—")

        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Profit factor",
            f"{report.profit_factor:.2f}" if report.profit_factor is not None else "—",
            help=(
                "Gross profit ÷ gross loss. Above 1.0 means the winners outweighed the losers; "
                "below 1.0 means they didn't. Says more than win rate, which ignores SIZE — a "
                "strategy can win 40% of the time and still be strongly profitable. "
                "'—' means undefined: either no trades, or no losing trades to divide by "
                "(which on a small sample usually means too few trades to judge, not perfection)."
            ),
        )
        p2.metric(
            "Time in market",
            f"{report.time_in_market:.1%}" if report.time_in_market is not None else "—",
            help=(
                "Share of the window's bars holding a position. Lower for the same return means "
                "your money was exposed to the market for less of the time."
            ),
        )
        p3.metric(
            "Avg trade value",
            f"${report.avg_trade_value:,.0f}" if report.avg_trade_value is not None else "—",
            help=(
                "Average capital committed per trade. This engine goes all-in on a BUY, so this "
                "tracks the equity curve — a sanity check on position size, not a strategy metric."
            ),
        )

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

    universe_name = st.selectbox("Universe", options=list(UNIVERSE_REGISTRY.keys()), key="scan_universe")
    # Sharpe annualization needs the right session-length/trading-days-per-year
    # pair for whichever asset class this universe is — derived from the
    # universe pick so it can never be forgotten per-scan the way a manual
    # checkbox could be. See metrics.MARKET_CALENDARS for why crypto (24/7,
    # 365 days) and forex (24h session but closed weekends, 252 days) are NOT
    # the same calendar despite both trading a "continuous" daily session.
    if universe_name.startswith("Crypto"):
        market_calendar = "crypto"
        st.caption(
            "Crypto trades 24/7 — Sharpe is annualized on a 365-day, 24-hour-session "
            "calendar instead of the equity default. No survivorship bias here (this is "
            "a fixed hand-picked list, not a historical index membership snapshot)."
        )
    elif universe_name.startswith("Forex"):
        market_calendar = "forex"
        st.caption(
            "Forex trades a 24-hour session but closes on weekends — Sharpe is annualized "
            "on a 252-trading-day year (like equities) with a 24-hour session (unlike "
            "equities), not the crypto 365-day calendar. No survivorship bias here (this is "
            "a fixed list of major pairs, not a historical index membership snapshot)."
        )
    else:
        market_calendar = "equity"
        st.caption(
            "⚠ Both universes below are today's constituent list applied retroactively over "
            "the historical window — this overstates performance somewhat, since companies "
            "removed/delisted from the index along the way aren't included (survivorship bias)."
        )

    try:
        universe_df = load_universe(universe_name)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    col1, col2 = st.columns(2)
    with col1:
        max_tickers = st.slider(
            f"Number of {universe_name} tickers to scan (evenly sampled across the whole list)",
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

    st.caption("Known risk-event step-out")
    event_filter_enabled = st.checkbox(
        "Skip new entries on FOMC meeting days — the proactive \"don't sell insurance during "
        "a flood warning\" filter (exits still fire). No extra API calls. Off by default so "
        "existing scan results stay comparable.",
        value=False, key="scan_event_filter",
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
        f"{multiplier}|{timespan}|{vol_target_enabled}|{target_vol_ann}|{event_filter_enabled}"
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

    tickers = sample_universe(universe_df, max_tickers)["ticker"].tolist()
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
                market_calendar=market_calendar,
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
            event_filter_enabled=event_filter_enabled,
            market_calendar=market_calendar,
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
        "engine_version": ENGINE_VERSION,
        "vol_target_enabled": vol_target_enabled,
        "target_vol_ann": target_vol_ann,
        "event_filter_enabled": event_filter_enabled,
    }
    paths = save_report(results, meta, results_dir)
    record_scan(meta, results)
    st.success("Results saved to Scan History — see the **Scan History** tab. A standalone report is also available below if you want one.")

    aggregates = aggregate_by_strategy(results)
    st.subheader("Per-strategy summary")
    st.dataframe(pd.DataFrame([asdict(a) for a in aggregates]), width="stretch")

    ranked = rank_combos(results)
    st.subheader("Top combos")
    st.caption(
        "Expectancy = what an average trade returned (win% × avg win + loss% × avg loss). "
        "PF = profit factor (gross profit ÷ gross loss; above 1 = winners outweighed losers). "
        "ER = the ticker's efficiency ratio over the window: low ≈ range-bound, high ≈ trending "
        "— match range strategies to low-ER tickers. All are informational; none affect the score."
    )
    st.dataframe(
        ranked.head(30),
        width="stretch",
        column_config={
            "expectancy": st.column_config.NumberColumn("Expectancy/trade", format="percent"),
            "profit_factor": st.column_config.NumberColumn(
                "PF", format="%.2f",
                help="Gross profit ÷ gross loss. Blank = undefined (no trades, or no losing trades).",
            ),
            "time_in_market": st.column_config.NumberColumn(
                "In market", format="percent",
                help="Share of the window's bars holding a position.",
            ),
            "efficiency_ratio": st.column_config.NumberColumn("ER", format="%.2f"),
        },
    )

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


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_account_balances(account_ids: tuple[str, ...], history_period: str) -> dict:
    """Cached so the Accounts tab's balance/history section only actually hits
    each broker once per 60s, not on every Streamlit rerun (any widget
    interaction anywhere on the page reruns the whole script, and this used
    to call build_broker_accounts + get_account_snapshot unconditionally every
    time balances were toggled on). IG specifically tolerates only one fresh
    login per short window (see ig.py's session-reuse notes) — an uncached
    per-rerun call here would keep re-triggering that. Returns plain
    JSON-safe data (not live broker objects), so st.cache_data can hash/store
    it; friendly error strings are computed here too, before the original
    exception goes out of scope."""
    result: dict = {"connect_error": None, "rows": [], "row_errors": [], "history": []}
    try:
        broker_accounts = build_broker_accounts(list(account_ids))
    except Exception as e:  # noqa: BLE001
        result["connect_error"] = str(e)
        return result

    pnl_by_account = realized_pnl_by_account()

    for broker_account in broker_accounts:
        try:
            snapshot = broker_account.get_account_snapshot()
            result["rows"].append(
                {
                    "account": broker_account.nickname,
                    "mode": "Paper" if broker_account.is_paper else "LIVE",
                    "equity": snapshot.equity,
                    "cash": snapshot.cash,
                    "buying_power": snapshot.buying_power,
                    # Realized P&L only (closed round trips from live_trades.db) —
                    # not unrealized/open-position P&L, which _render_live_positions
                    # already shows separately per open position.
                    "realized_pnl": pnl_by_account.get(broker_account.account_id, 0.0),
                }
            )
        except Exception as e:  # noqa: BLE001
            result["row_errors"].append((broker_account.nickname, "balance", _friendly_account_error(e)))
            continue

        try:
            points = broker_account.get_equity_history(period=history_period, timeframe="1D")
            if points:
                result["history"].append(
                    {
                        "name": broker_account.nickname,
                        "x": [p.timestamp for p in points],
                        "y": [p.equity for p in points],
                    }
                )
        except Exception as e:  # noqa: BLE001
            result["row_errors"].append((broker_account.nickname, "equity history", _friendly_account_error(e)))

    return result


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

    def _render_account_rows(accounts: list[dict]) -> None:
        hdr = st.columns([3, 2, 2, 3, 1])
        hdr[0].caption("Account")
        hdr[1].caption("Broker")
        hdr[2].caption("Mode")
        hdr[3].caption("Market")
        for acct in accounts:
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

    if linked:
        # Grouped by what each linked account actually trades (not just its
        # broker type — IBKR alone can be either) so an equity/forex/crypto
        # mismatch like the Trade Execution one is visible here too, not just
        # prevented there. Fixed section order; empty sections render nothing.
        sections = [
            ("Equities", "equity"),
            ("Forex & CFDs", "forex"),
            ("Crypto", "crypto"),
        ]
        for label, asset_class in sections:
            group = [a for a in linked if account_asset_class(a) == asset_class]
            if not group:
                continue
            st.markdown(f"**{label}**")
            _render_account_rows(group)
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
            fetched = _fetch_account_balances(tuple(a["id"] for a in linked), history_period)

            if fetched["connect_error"]:
                st.error(f"Failed to connect to linked accounts: {fetched['connect_error']}")

            for nickname, kind, err in fetched["row_errors"]:
                if kind == "balance":
                    st.info(f"{nickname}: balance {err}")
                else:
                    st.caption(f"{nickname}: equity history {err}")

            balance_rows = fetched["rows"]
            if balance_rows:
                display_df = pd.DataFrame(balance_rows).rename(columns={"realized_pnl": "Realized P&L"})
                st.dataframe(display_df, use_container_width=True)
                total_pnl = sum(r["realized_pnl"] for r in balance_rows)
                sign = "-" if total_pnl < 0 else "+"
                st.metric(
                    "Total realized P&L (all accounts)",
                    f"${total_pnl:,.2f}",
                    delta=f"{sign}${abs(total_pnl):,.2f}",
                )
                st.caption(
                    "Realized P&L = closed round trips only (from live_trades.db), not unrealized "
                    "P&L on currently open positions — see the Trade Execution tab for those."
                )

            history_fig = go.Figure()
            for series in fetched["history"]:
                history_fig.add_trace(
                    go.Scatter(x=series["x"], y=series["y"], mode="lines", name=series["name"])
                )

            if fetched["history"]:
                history_fig.update_layout(
                    title="Account equity over time (all linked accounts)",
                    xaxis_title="Date",
                    yaxis_title="Equity ($)",
                    height=380,
                )
                st.plotly_chart(history_fig, use_container_width=True)
            elif balance_rows:
                st.caption("No balance history available yet for the selected range.")
            st.caption("Balances are cached for 60s.")

    if linked:
        st.caption("Live positions with colour-coded P&L are now on the **Overview** page.")

    st.divider()
    st.subheader("Link a new account")
    broker = st.selectbox(
        "Broker", SUPPORTED_BROKERS, format_func=lambda b: BROKER_META[b]["label"], key="link_broker"
    )
    broker_meta = BROKER_META[broker]
    uses_gateway = broker_meta.get("uses_gateway", False)
    extra_cred_label = broker_meta.get("extra_cred_field")
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

        api_key = secret_key = extra_cred = ""
        ibkr_host = ibkr_account = ""
        ibkr_port, ibkr_client_id = 4002, 1
        ibkr_asset_class = "Equity"

        if uses_gateway:
            # IBKR connects to a local gateway the user runs and logs into — no API
            # key/secret, just connection config. Rendered as normal (non-masked) inputs.
            st.caption(
                "Interactive Brokers connects to a local **IB Gateway** (or Trader Workstation) that "
                "you run and log into — there's no API key to enter here. Start the gateway, enable its "
                "API (Configure → Settings → API → Enable ActiveX and Socket Clients), and point this at "
                "its host/port. Ports: IB Gateway **4002 paper / 4001 live** (TWS 7497 / 7496)."
            )
            ibkr_asset_class = st.radio(
                "Asset class this account trades",
                ["Equity", "Forex"],
                index=0,
                horizontal=True,
                help="Set once here, matching how you've configured this IBKR account/permissions on "
                "IBKR's side. Fixed for the account's lifetime in this dashboard — unlink and re-link "
                "to change it. A single IBKR login can have separate equity and forex sub-accounts; "
                "link each one separately if you trade both.",
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
        elif extra_cred_label:
            # IG-shaped brokers: 3 credentials, not the usual 2 — its own
            # account PASSWORD stands in for a "secret", plus a username IG
            # needs to identify the session (see accounts.py's
            # extra_cred_field / add_account's extra_cred param).
            st.caption(
                f"{broker_meta['label']} doesn't use a key+secret pair — it needs your "
                f"account username and password alongside the API key you generated on "
                f"their platform."
            )
            extra_cred = st.text_input(extra_cred_label, type="password")
            api_key = st.text_input(cred_label_1, type="password")
            secret_key = st.text_input(cred_label_2, type="password")
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
                            "asset_class": "forex" if ibkr_asset_class == "Forex" else "equity",
                        },
                    )
                    st.success(f"Linked {nickname}.")
                    st.rerun()
            elif extra_cred_label:
                if not nickname or not api_key or not secret_key or not extra_cred:
                    st.error(f"Nickname, {extra_cred_label}, {cred_label_1}, and {cred_label_2} are all required.")
                elif not is_paper and not live_confirm:
                    st.error("Check the confirmation box to link this account.")
                else:
                    add_account(nickname, broker, is_paper, api_key, secret_key, extra_cred=extra_cred)
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

    exec_asset_class_label = st.radio(
        "Asset class", ["Equity", "Crypto", "Forex"], horizontal=True, key="exec_asset_class",
        help="Scopes both the ticker list and the target-account list below to only "
        "compatible combinations — an equity account can't be offered a forex ticker "
        "(or vice versa), the mismatch that used to only get caught by the broker itself.",
    )
    asset_class_key = {"Equity": "equity", "Crypto": "crypto", "Forex": "forex"}[exec_asset_class_label]
    # Keying the ticker widget per asset class forces a true remount when the
    # radio changes, instead of reusing one widget instance whose displayed
    # text otherwise doesn't reliably reset via session_state alone (observed
    # live: popping session_state["exec_ticker"] left the combobox showing a
    # stale equity ticker after switching to Forex, even though its options
    # list had correctly narrowed) — a stale LABEL only, not a stale VALUE
    # (the "Submit order" re-check below still catches a genuine mismatch),
    # but confusing enough to fix properly rather than leave as a footnote.
    ticker_widget_key = f"exec_ticker_{asset_class_key}"

    known = _known_tickers_for_asset_class(asset_class_key)
    ticker_options = sorted(known)
    # A hand-typed symbol not in this asset class's universe CSV (e.g. a raw
    # IG epic) may still be valid for the selected broker — accept_new_options
    # keeps free-text entry available, same as the Backtest tab's ticker picker.
    current_ticker = st.session_state.get(ticker_widget_key)
    if current_ticker and current_ticker not in known:
        ticker_options = [current_ticker] + ticker_options
    default_ticker = {"equity": "AAPL", "crypto": "X:BTCUSD", "forex": "C:EURUSD"}[asset_class_key]
    ticker_choice = st.selectbox(
        "Ticker (type to search, or enter any symbol not listed)",
        options=ticker_options,
        index=ticker_options.index(default_ticker) if default_ticker in ticker_options else 0,
        format_func=lambda t: f"{t} — {known[t]}" if known.get(t) else t,
        key=ticker_widget_key,
        accept_new_options=True,
    )
    ticker = (ticker_choice or default_ticker).upper().strip()
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

    compatible_accounts = [a for a in linked if account_asset_class(a) == asset_class_key]
    account_options = {
        f"{a['nickname']} ({'Paper' if a['is_paper'] else 'LIVE'})": a["id"] for a in compatible_accounts
    }
    if not compatible_accounts:
        st.info(
            f"No linked accounts trade {exec_asset_class_label}. Link one in the **Accounts** tab first."
        )
        return
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
        # Defense-in-depth: account_orders was captured at "Preview order" time
        # and lives in session_state, so it can go stale if the Asset class
        # radio or ticker changes after previewing but before submitting (a
        # full rerun narrows compatible_accounts/the ticker list immediately,
        # but doesn't retroactively invalidate an already-built preview).
        # Re-check every previewed account against the CURRENT selection
        # before anything reaches a broker — mirrors the belt-and-suspenders
        # check auto_trader.py's _trade_target already does per cycle.
        compatible_ids = {a["id"] for a in compatible_accounts}
        stale_orders = [
            ao for ao in account_orders
            if ao.account.account_id not in compatible_ids or infer_asset_class(ticker) != asset_class_key
        ]
        if stale_orders:
            st.error(
                "The ticker or asset class changed since this order was previewed — click "
                "**Preview order** again to refresh it before submitting."
            )
            st.stop()
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

    regime_match_only = st.checkbox(
        "Only promote regime-matched combos — pair range strategies (mean reversion) with "
        "range-bound tickers and trend strategies with trending ones, measured by each "
        "ticker's efficiency ratio in the scan. Off by default (measure first).",
        value=state.config.regime_match_only, key="roster_regime_match",
    )

    st.caption("Diversification caps (0 = no limit) — applied when you re-evaluate the roster")
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        max_per_strategy = st.number_input(
            "Max active combos per strategy", min_value=0, value=state.config.max_per_strategy,
            key="roster_max_per_strategy",
            help="Stops one strategy filling every slot. If its edge breaks, only part of the "
                 "roster is affected instead of all of it.",
        )
    with dcol2:
        max_per_sector = st.number_input(
            "Max active combos per sector", min_value=0, value=state.config.max_per_sector,
            key="roster_max_per_sector",
            help="Stops the roster concentrating in one sector. Tickers with an unknown sector "
                 "are never blocked.",
        )

    new_config = roster.RosterConfig(
        roster_size=int(roster_size),
        min_live_trades=int(min_live_trades),
        losing_streak_threshold=int(losing_streak_threshold),
        win_rate_floor=win_rate_floor / 100,
        cum_pnl_floor=cum_pnl_floor,
        max_pnl_drawdown_floor=max_pnl_drawdown_floor,
        weights=state.config.weights,
        regime_match_only=regime_match_only,
        max_per_strategy=int(max_per_strategy),
        max_per_sector=int(max_per_sector),
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
            ticker_regime = classify_ticker_regime(e.efficiency_ratio)
            strat_regime = strategy_regime(e.strategy_name)
            if e.efficiency_ratio is None:
                match = "—"
            elif "either" in (ticker_regime, strat_regime) or ticker_regime == strat_regime:
                match = f"✅ {strat_regime} / {ticker_regime}"
            else:
                match = f"⚠️ {strat_regime} / {ticker_regime}"
            rows.append(
                {
                    "ticker": e.ticker,
                    "sector": sector_for_ticker(e.ticker) or "—",
                    "strategy": e.strategy_name,
                    "status": e.status,
                    "backtest_score": round(e.backtest_score, 3),
                    "regime match": match,
                    "ER": round(e.efficiency_ratio, 2) if e.efficiency_ratio is not None else None,
                    "live_trades": live.get("num_trades"),
                    "live_win_rate": f"{live['win_rate']:.0%}" if live.get("win_rate") is not None else "—",
                    "live_total_pnl": live.get("total_pnl"),
                    "losing_streak": live.get("current_losing_streak"),
                    "reason": e.pause_reason or "",
                }
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Regime match compares the strategy's design (trend/range) against the ticker's "
            "measured behaviour over the scan window. ⚠️ flags a mismatch — informational unless "
            "the regime-match setting above is on. Blank for entries from scans before this existed."
        )

        # Concentration readout: the roster that actually trades is the ACTIVE
        # set, so measure that rather than the whole candidate list.
        active_entries = [e for e in state.entries if e.status == "active"]
        if active_entries:
            strat_counts = Counter(e.strategy_name for e in active_entries)
            sector_counts = Counter(sector_for_ticker(e.ticker) or "unknown" for e in active_entries)
            top_strat, top_strat_n = strat_counts.most_common(1)[0]
            top_sector, top_sector_n = sector_counts.most_common(1)[0]
            total = len(active_entries)
            summary = (
                f"**Active roster concentration:** {total} combo(s) across "
                f"{len(strat_counts)} strategy(s) and {len(sector_counts)} sector(s). "
                f"Largest: {top_strat_n}/{total} on *{top_strat}*, {top_sector_n}/{total} in *{top_sector}*."
            )
            if total > 1 and (top_strat_n == total or top_sector_n == total):
                st.warning(
                    summary + " Every active slot shares a strategy or sector — one broken edge "
                    "would hit the whole roster at once. Consider lowering the caps above and "
                    "re-evaluating."
                )
            else:
                st.caption(summary)

    st.markdown("#### Promotion / demotion history")
    events = roster.load_events(limit=50)
    if not events:
        st.caption("No promotions or demotions recorded yet.")
    else:
        st.dataframe(pd.DataFrame(events), width="stretch", hide_index=True)

    st.divider()


# Conservative/aggressive risk dial (CLAUDE_NOTES.txt "risk dial" entry). Only
# ever touches the sizing/risk-LIMIT knobs (axis 2b) — never the strategies'
# own entry-selectivity params (axis 2a), which are a separate, already-
# answered question (Phase 1-4's own tuning). vol_target_ann values are taken
# straight from the real overnight sweep's tested grid ([10, 15, 20, 25, 30]),
# not interpolated — Conservative/Moderate/Aggressive map to the sweep's own
# low/default/high points so each preset is backed by an actual measured
# result, not a guess. All three enable vol-target sizing (the sweep only
# ever tested it ON); Moderate matches today's status-quo defaults exactly,
# so applying it is a no-op for an account already running the defaults.
RISK_PRESETS = {
    "Conservative": {
        "sizing_value": 0.5, "vol_target_ann": 10.0,
        "max_drawdown_pct": 7.0, "giveback_enabled": True, "giveback_pct": 15.0,
    },
    "Moderate": {
        "sizing_value": 1.0, "vol_target_ann": 20.0,
        "max_drawdown_pct": 10.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
    "Aggressive": {
        "sizing_value": 2.0, "vol_target_ann": 30.0,
        "max_drawdown_pct": 15.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
}


def _apply_risk_preset(name: str) -> None:
    """Load-mutate-save control.json (single source of truth) AND push the
    same values into every affected widget's session_state key directly —
    those widgets already have a `key=`, so Streamlit ignores `value=` on
    later reruns once a key exists (same reasoning as the execution-cost
    preset buttons above). Spans two tabs (sizing/vol-target live on this
    page, max-drawdown/giveback live on Settings) — session_state is global
    across st.navigation pages, so this correctly updates Settings too, even
    though the button is here."""
    preset = RISK_PRESETS[name]
    control = load_control()
    control.sizing_mode = SizingMode.PCT_EQUITY.value
    control.sizing_value = preset["sizing_value"]
    control.vol_target_enabled = True
    control.vol_target_ann = preset["vol_target_ann"]
    control.max_drawdown_enabled = True
    control.max_drawdown_pct = preset["max_drawdown_pct"]
    control.giveback_enabled = preset["giveback_enabled"]
    control.giveback_pct = preset["giveback_pct"]
    save_control(control)

    st.session_state["auto_sizing_mode"] = "% of account equity"
    st.session_state["auto_sizing_value"] = preset["sizing_value"]
    st.session_state["auto_vol_target"] = True
    st.session_state["auto_vol_target_ann"] = preset["vol_target_ann"]
    st.session_state["settings_max_dd_enabled"] = True
    st.session_state["settings_max_dd_pct"] = preset["max_drawdown_pct"]
    st.session_state["settings_giveback_enabled"] = preset["giveback_enabled"]
    st.session_state["settings_giveback_pct"] = preset["giveback_pct"]
    st.session_state["risk_preset_applied"] = name


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

    st.markdown("**Risk profile**")
    st.caption(
        "Sets position sizing (% of equity), the GARCH vol-target, the account-level max-"
        "drawdown breaker, and the daily P&L giveback guard together — the sizing/risk-LIMIT "
        "knobs only, never the strategies' own entry logic (that's already tuned separately, "
        "see CLAUDE_NOTES.txt). vol-target values come straight from the real overnight sweep "
        "(2026-08-07/08): across every strategy with a genuine edge, higher sizing never hurt "
        "risk-adjusted returns within the 10-30% range tested — it only ever helped or was "
        "neutral. **Moderate matches today's defaults exactly** — applying it is a safe no-op on "
        "an account already running as-is. Also updates the Settings page's account-risk and "
        "giveback sections."
    )
    applied_preset = st.session_state.pop("risk_preset_applied", None)
    if applied_preset is not None:
        st.success(
            f"**{applied_preset}** applied and saved — sizing/vol-target below and the "
            "account-risk/giveback sections on the Settings page all updated together."
        )
    pcol1, pcol2, pcol3 = st.columns(3)
    for pcol, name in zip((pcol1, pcol2, pcol3), RISK_PRESETS):
        with pcol:
            preset = RISK_PRESETS[name]
            st.button(
                name, key=f"risk_preset_{name}", width="stretch",
                on_click=_apply_risk_preset, args=(name,),
                help=(
                    f"{preset['sizing_value']:.1f}% of equity/trade · "
                    f"{preset['vol_target_ann']:.0f}% vol target · "
                    f"{preset['max_drawdown_pct']:.0f}% max drawdown · "
                    f"giveback guard {'on 15%' if preset['giveback_enabled'] else 'off'}"
                ),
            )

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

        st.caption("Known risk-event step-out")
        block_event_days = st.checkbox(
            "Skip new BUYs on FOMC meeting days (exits unaffected) — step out of events that are "
            "known in advance to move the market, instead of waiting for volatility to show up in "
            "the data. On by default.",
            value=control.block_event_days, key="auto_block_events",
        )

    st.markdown("**Target accounts**")
    st.caption(
        "Toggle each linked account independently — test one broker at a time, or several "
        "together, without having to remember to remove others first. Grouped by what each "
        "account trades — a ticker only ever gets attempted against a matching account "
        "(auto_trader.py's own asset-class guard), regardless of what's checked here."
    )
    selected_ids = []
    sizing_overrides: dict[str, dict] = {}
    for section_label, asset_class in [("Equities", "equity"), ("Forex & CFDs", "forex"), ("Crypto", "crypto")]:
        group = [a for a in linked if account_asset_class(a) == asset_class]
        if not group:
            continue
        st.caption(section_label)
        for a in group:
            broker_label = BROKER_META.get(a["broker"], {}).get("label", a["broker"])
            mode_label = "Paper" if a["is_paper"] else "LIVE"
            checked = st.checkbox(
                f"{a['nickname']} — {broker_label} ({mode_label})",
                value=a["id"] in control.account_ids,
                key=f"auto_account_{a['id']}",
            )
            if checked:
                selected_ids.append(a["id"])
                existing_override = control.account_sizing_overrides.get(a["id"], {})
                ocol1, ocol2 = st.columns(2)
                with ocol1:
                    below_equity = st.number_input(
                        f"↳ {a['nickname']}: small-account sizing while equity is below ($, 0 = disabled)",
                        min_value=0.0,
                        value=float(existing_override.get("below_equity", 0.0)),
                        step=10.0,
                        key=f"auto_sizing_threshold_{a['id']}",
                    )
                with ocol2:
                    fixed_dollars = st.number_input(
                        f"↳ {a['nickname']}: fixed $/trade while below that",
                        min_value=0.0,
                        value=float(existing_override.get("fixed_dollars", 0.0)),
                        step=1.0,
                        key=f"auto_sizing_fixed_{a['id']}",
                    )
                if below_equity > 0 and fixed_dollars > 0:
                    sizing_overrides[a["id"]] = {"below_equity": below_equity, "fixed_dollars": fixed_dollars}
                elif below_equity > 0 or fixed_dollars > 0:
                    st.warning(
                        f"{a['nickname']}: set BOTH the threshold and the fixed $/trade for the "
                        "override to apply — half-set values are ignored."
                    )
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
            account_sizing_overrides=sizing_overrides,
            vol_target_enabled=vol_target_enabled,
            vol_target_ann=vol_target_ann,
            block_event_days=block_event_days,
            use_roster=use_roster,
            # Owned by the Settings page now — carry forward from the loaded control so
            # saving here never clobbers the account-risk limit set in Settings.
            max_drawdown_enabled=control.max_drawdown_enabled,
            max_drawdown_pct=control.max_drawdown_pct,
            # Same reasoning: the daily P&L giveback guard is also Settings-owned.
            giveback_enabled=control.giveback_enabled,
            giveback_pct=control.giveback_pct,
            # And the "Extra target" section below owns these two — carry them
            # forward so saving ordinary settings here never wipes it.
            manual_strategy_params=control.manual_strategy_params,
            extra_targets=control.extra_targets,
        )
        save_control(new_control)
        st.success("Configuration saved.")
        st.rerun()

    st.divider()
    st.markdown("### Extra targets (optional)")
    st.caption(
        "Additional CONCURRENT targets that trade alongside everything above — e.g. a forex "
        "account running its own separately-tuned strategy while the primary (roster or manual) "
        "trades equities. Any number of these can run at once, each with its own account/ticker/"
        "strategy/params — but they all still share the primary's daily trade cap, sizing, and "
        "bar interval (not independently configurable yet), and are evaluated AFTER the primary "
        "each cycle in order, so the last one in the list is what gets skipped first on a day "
        "the shared cap fills early."
    )

    if control.extra_targets:
        for i, et in enumerate(control.extra_targets):
            et_accounts = ", ".join(
                next((a["nickname"] for a in linked if a["id"] == aid), aid)
                for aid in et.get("account_ids", [])
            )
            with st.expander(et.get("label") or f"Extra target {i + 1}"):
                st.write(f"Accounts: {et_accounts}")
                st.write(f"Tickers: {', '.join(et.get('tickers', []))}")
                st.write(f"Strategy: {et.get('strategy_name')} — params: {et.get('strategy_params', {})}")
                if st.button("Remove this extra target", key=f"remove_extra_{i}"):
                    fresh = load_control()
                    fresh.extra_targets = [t for j, t in enumerate(fresh.extra_targets) if j != i]
                    save_control(fresh)
                    st.rerun()
    else:
        st.caption("None configured yet.")

    st.markdown("**Add a new extra target**")
    existing_extra: dict = {}  # the form below always builds a NEW entry, never edits one in place

    extra_asset_class_label = st.radio(
        "Asset class", ["Equity", "Crypto", "Forex"], horizontal=True, key="extra_asset_class",
    )
    extra_asset_class_key = {"Equity": "equity", "Crypto": "crypto", "Forex": "forex"}[extra_asset_class_label]

    extra_accounts_in_class = [a for a in linked if account_asset_class(a) == extra_asset_class_key]
    extra_account_options = {
        f"{a['nickname']} — {BROKER_META.get(a['broker'], {}).get('label', a['broker'])} "
        f"({'Paper' if a['is_paper'] else 'LIVE'})": a["id"]
        for a in extra_accounts_in_class
    }
    if not extra_account_options:
        st.info(f"No linked accounts trade {extra_asset_class_label}. Link one in the **Accounts** tab first.")
    extra_selected_labels = st.multiselect(
        "Extra target accounts",
        options=list(extra_account_options.keys()),
        default=[
            label for label, aid in extra_account_options.items()
            if aid in existing_extra.get("account_ids", [])
        ],
        key="extra_target_accounts",
    )
    extra_account_ids = [extra_account_options[label] for label in extra_selected_labels]

    extra_known = _known_tickers_for_asset_class(extra_asset_class_key)
    extra_tickers = st.multiselect(
        "Extra target tickers",
        options=sorted(extra_known),
        default=[t for t in existing_extra.get("tickers", []) if t in extra_known],
        format_func=lambda t: f"{t} — {extra_known[t]}" if extra_known.get(t) else t,
        key="extra_target_tickers",
    )

    extra_strategy_options = list(STRATEGY_REGISTRY.keys())
    extra_strategy_default_idx = (
        extra_strategy_options.index(existing_extra["strategy_name"])
        if existing_extra.get("strategy_name") in extra_strategy_options else 0
    )
    extra_strategy_name = st.selectbox(
        "Extra target strategy", options=extra_strategy_options,
        index=extra_strategy_default_idx, key="extra_target_strategy",
    )

    extra_params_text = st.text_area(
        "Strategy params override (JSON, merged over the strategy's normal defaults — leave as "
        "{} to use the defaults)",
        value=json.dumps(existing_extra.get("strategy_params", {})),
        key="extra_target_params_json",
        help='e.g. {"entry_deviation_pct": 0.2} for VWAP Mean Reversion, or {"num_std": 4.0, '
        '"period": 20} for Bollinger Mean Reversion — the forex-tuned values validated '
        "separately from the equities defaults, not achievable any other way today.",
    )

    extra_label = st.text_input(
        "Label (for your own reference only)",
        value=existing_extra.get("label", ""), key="extra_target_label",
    )

    if st.button("Add extra target"):
        try:
            extra_params = json.loads(extra_params_text) if extra_params_text.strip() else {}
            if not isinstance(extra_params, dict):
                raise ValueError('must be a JSON object, e.g. {"param": value}')
        except (json.JSONDecodeError, ValueError) as e:
            st.error(f"Strategy params override isn't valid JSON: {e}")
        else:
            if extra_account_ids and extra_tickers and extra_strategy_name:
                # Load-mutate-save so only extra_targets changes — never clobber
                # the primary config this same tab's other Save button owns, and
                # APPEND rather than replace so this doesn't wipe out any extra
                # targets already configured (e.g. an existing forex one) when
                # adding another.
                fresh = load_control()
                fresh.extra_targets = fresh.extra_targets + [{
                    "label": extra_label or f"{extra_asset_class_label} extra target",
                    "account_ids": extra_account_ids,
                    "tickers": extra_tickers,
                    "strategy_name": extra_strategy_name,
                    "strategy_params": extra_params,
                }]
                save_control(fresh)
                st.success("Extra target added.")
                st.rerun()
            else:
                st.error("Pick at least one account, one ticker, and a strategy before adding.")

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
            "mean_expectancy": st.column_config.NumberColumn("Mean expectancy", format="percent", help="Average per-trade expectancy (win% × avg win + loss% × avg loss) across this strategy's results. Positive means an average trade made money."),
            "mean_profit_factor": st.column_config.NumberColumn("Mean PF", format="%.2f", help="Average profit factor (gross profit ÷ gross loss) across this strategy's results. This averages per-combo RATIOS, so a combo with a huge ratio off two or three trades can drag it upward — read it next to the Results count, as a signpost rather than a measurement."),
            "mean_time_in_market": st.column_config.NumberColumn("Mean in market", format="percent", help="Average share of each window's bars spent holding a position."),
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
            "expectancy": st.column_config.NumberColumn("Expectancy/trade", format="percent", help="Win% × avg win + loss% × avg loss — what an average trade returned."),
            "profit_factor": st.column_config.NumberColumn("PF", format="%.2f", help="Profit factor: gross profit ÷ gross loss. Above 1 means the winners outweighed the losers. Blank = undefined (no trades, or no losing trades to divide by)."),
            "time_in_market": st.column_config.NumberColumn("In market", format="percent", help="Share of the scan window's bars this combo spent holding a position."),
            "efficiency_ratio": st.column_config.NumberColumn("ER", format="%.2f", help="Ticker's efficiency ratio over the scan window: low (<0.25) ≈ range-bound, high (>0.35) ≈ trending. Match range strategies to low-ER tickers."),
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


def _launch_scan_runner_process() -> None:
    kwargs = {"cwd": str(PROJECT_ROOT)}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    proc = subprocess.Popen([sys.executable, str(SCAN_RUNNER_SCRIPT)], **kwargs)
    st.session_state["scan_runner_proc"] = proc


def render_playlist_page() -> None:
    st.subheader("Backtest playlist")
    st.caption(
        "Queue several scans and let them run unattended in a separate process — it keeps "
        "going after you close this tab. Results land in **Scan history** exactly like a "
        "manual scan. This is the bulk data-gathering path: broad, varied backtests build the "
        "dataset that future learning features train on."
    )

    items = playlist.load_playlist()
    status = playlist.load_status()

    heartbeat_stale = True
    if status.last_heartbeat:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
            heartbeat_stale = age > 900  # a ticker can legitimately take minutes at 5 req/min
        except ValueError:
            heartbeat_stale = True
    runner_alive = status.running and not heartbeat_stale

    pending_count = sum(1 for i in items if i.status == playlist.PENDING)
    done_count = sum(1 for i in items if i.status == playlist.DONE)

    st.markdown("### Runner")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runner", "🟢 Running" if runner_alive else "🔴 Not running")
    m2.metric("Queued", pending_count)
    m3.metric("Completed", done_count)
    m4.metric("Failed", sum(1 for i in items if i.status == playlist.FAILED))

    if runner_alive and status.current_label:
        st.progress(
            min(max(status.current_progress, 0.0), 1.0),
            text=f"{status.current_label} — {status.current_ticker or 'starting'}",
        )
    if status.running and heartbeat_stale:
        st.warning(
            "The runner is flagged as running but hasn't sent a heartbeat in a while — it was "
            "probably killed (laptop sleep, reboot). Starting again is safe: finished tickers "
            "are checkpointed and won't be re-fetched."
        )
    if status.last_error:
        st.error(f"Last error: {status.last_error}")

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        if st.button("▶ Start runner", type="primary", disabled=runner_alive or pending_count == 0):
            playlist.request_stop(False)
            _launch_scan_runner_process()
            st.success("Runner started. It works through the queue and exits when it's empty.")
            time.sleep(1.5)
            st.rerun()
    with rc2:
        if st.button("⏹ Stop after current ticker", disabled=not runner_alive):
            playlist.request_stop(True)
            st.info("Stop requested — the runner finishes the current ticker, then stops. "
                    "The in-progress scan goes back to queued and resumes from its checkpoint.")
            st.rerun()
    with rc3:
        if st.button("🔄 Refresh"):
            st.rerun()

    st.divider()
    st.markdown("### Queue")
    if not items:
        st.caption("Nothing queued yet — add a scan below.")
    else:
        icons = {playlist.PENDING: "⏳", playlist.RUNNING: "▶️", playlist.DONE: "✅",
                 playlist.FAILED: "❌", playlist.CANCELLED: "⛔"}
        for pos, item in enumerate(items):
            with st.container(border=True):
                head, actions = st.columns([5, 2])
                with head:
                    st.markdown(f"{icons.get(item.status, '•')} **{item.describe()}**")
                    bits = [f"status: `{item.status}`"]
                    if item.run_id:
                        bits.append(f"scan run #{item.run_id}")
                    if item.num_results is not None:
                        bits.append(f"{item.num_results} results")
                    if item.vol_target_enabled:
                        bits.append("GARCH on")
                    if item.event_filter_enabled:
                        bits.append("event step-out on")
                    if item.position_mode != "long_only":
                        bits.append(f"mode: {item.position_mode}")
                    st.caption(" · ".join(bits))
                    if item.error:
                        st.error(item.error)
                with actions:
                    a1, a2, a3 = st.columns(3)
                    if a1.button("↑", key=f"up_{item.id}", disabled=pos == 0,
                                 help="Move earlier in the queue"):
                        playlist.move_item(item.id, -1)
                        st.rerun()
                    if a2.button("↓", key=f"down_{item.id}", disabled=pos == len(items) - 1,
                                 help="Move later in the queue"):
                        playlist.move_item(item.id, 1)
                        st.rerun()
                    if a3.button("🗑", key=f"del_{item.id}", disabled=item.status == playlist.RUNNING,
                                 help="Remove from the queue"):
                        playlist.remove_item(item.id)
                        st.rerun()
                    if item.status in (playlist.DONE, playlist.FAILED):
                        if st.button("Re-queue", key=f"requeue_{item.id}"):
                            playlist.reset_item(item.id)
                            st.rerun()

        if any(i.status in playlist.TERMINAL for i in items) and st.button("Clear finished items"):
            playlist.clear_finished()
            st.rerun()

    st.divider()
    st.markdown("### Add a scan to the queue")
    costs = app_settings.load_settings()
    with st.form("playlist_add"):
        label = st.text_input("Label (optional)", placeholder="e.g. 'S&P 500 · 3 months · hourly'")
        c1, c2 = st.columns(2)
        with c1:
            universe_name = st.selectbox("Universe", options=list(UNIVERSE_REGISTRY.keys()))
            max_tickers = st.number_input("Number of tickers", min_value=1, max_value=503, value=25)
            multiplier = st.number_input("Bar multiplier", min_value=1, value=1)
            timespan = st.selectbox("Bar unit", ["minute", "hour", "day"])
        with c2:
            from_date = st.date_input("From", value=date.today() - timedelta(days=30))
            to_date = st.date_input("To", value=date.today())
            requests_per_minute = st.number_input(
                "Polygon requests/minute", min_value=1, value=5,
                help="Match your Polygon plan. The runner honours this for every item.",
            )
            max_workers = st.number_input("Worker threads", min_value=1, max_value=16, value=4)
        strategy_names = st.multiselect(
            "Strategies", options=list(STRATEGY_REGISTRY.keys()),
            default=list(STRATEGY_REGISTRY.keys()),
        )
        f1, f2 = st.columns(2)
        with f1:
            vol_target_enabled = st.checkbox("GARCH volatility filter + sizing (doubles API calls)")
            target_vol_ann = st.number_input("Target annualized vol (%)", min_value=1.0, value=20.0)
        with f2:
            event_filter_enabled = st.checkbox("Skip entries on FOMC days")
            starting_cash = st.number_input("Starting cash ($)", min_value=100.0, value=100_000.0, step=1000.0)

        submitted = st.form_submit_button("Add to queue", type="primary")
        if submitted:
            if not strategy_names:
                st.error("Select at least one strategy.")
            elif from_date >= to_date:
                st.error("'From' must be before 'To'.")
            else:
                playlist.add_item(playlist.PlaylistItem(
                    label=label.strip(),
                    universe=universe_name,
                    max_tickers=int(max_tickers),
                    strategy_names=strategy_names,
                    from_date=str(from_date),
                    to_date=str(to_date),
                    multiplier=int(multiplier),
                    timespan=timespan,
                    starting_cash=starting_cash,
                    commission_per_trade=costs["commission"],
                    slippage_bps=costs["slippage_bps"],
                    requests_per_minute=int(requests_per_minute),
                    max_workers=int(max_workers),
                    vol_target_enabled=vol_target_enabled,
                    target_vol_ann=target_vol_ann,
                    event_filter_enabled=event_filter_enabled,
                ))
                st.success("Added to the queue.")
                st.rerun()

    calls = sum(i.max_tickers * (2 if i.vol_target_enabled else 1)
                for i in items if i.status == playlist.PENDING)
    if calls:
        slowest = min((i.requests_per_minute for i in items if i.status == playlist.PENDING), default=5)
        st.caption(
            f"Queued work: ~{calls} API calls at {slowest}/min ≈ {calls / slowest / 60:.1f} hours "
            "minimum, before per-ticker backtest compute. Scans are checkpointed, so stopping "
            "and restarting doesn't lose progress."
        )


def _format_age(seconds: float | None) -> str:
    """'3 minutes ago' / '29 hours ago'. Deliberately coarse — the exact second a
    dead process last spoke doesn't matter, the order of magnitude does."""
    if seconds is None:
        return "at an unknown time"
    if seconds < 90:
        return f"{int(seconds)} seconds ago"
    if seconds < 5400:
        return f"{round(seconds / 60)} minutes ago"
    if seconds < 172800:
        return f"{round(seconds / 3600)} hours ago"
    return f"{round(seconds / 86400)} days ago"


def _render_bot_status(status, control, *, kill_switch: bool = True, kill_key: str = "kill_switch_btn") -> bool:
    """Auto-trader status row (process / armed / trades today / killed) plus the
    last signal/error and an optional kill switch. Shared by the Overview home
    and the Auto Trading tab so both always show the same truth. Returns whether
    the process is actually running (heartbeat-fresh), which the Auto Trading tab
    uses to decide whether to launch a new process on Start."""
    heartbeat_stale = True
    age = None
    if status.last_heartbeat:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
            heartbeat_stale = age > max(control.poll_interval_seconds * 3, 30)
        except ValueError:
            heartbeat_stale = True
            age = None
    actually_running = status.running and not heartbeat_stale

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Process", "🟢 Running" if actually_running else "🔴 Not running")
    c2.metric("Armed", "🟢 Yes" if control.enabled and not control.killed else "🔴 No")
    c3.metric("Trades today", f"{status.trades_today} / {control.max_trades_per_day}")
    c4.metric("Killed", "⚠ Yes" if control.killed else "No")

    # A bot that STOPPED itself sets running=False on the way out. A stale heartbeat
    # while running is still True means the process vanished without cleaning up —
    # sleep, crash, power loss — which is exactly the failure that went unnoticed for
    # ~29 hours on 22 July. Say so loudly and differently from a deliberate stop.
    if status.running and heartbeat_stale:
        st.error(
            f"**Bot stopped unexpectedly.** It last checked in {_format_age(age)} — the process "
            "is gone but never recorded a clean shutdown, which usually means the laptop slept "
            "or the process was killed. Any open position is currently unmanaged."
        )
        if not heartbeat.load_config().get("enabled"):
            st.warning(
                "You'd only have found this by looking. Turn on the bot-down alert in "
                "**Settings** to get paged when this happens away from the screen."
            )
    elif control.enabled and not control.killed and not actually_running:
        st.warning(
            "**Armed, but nothing is running.** The settings say trade, but there's no live "
            "process to act on them — no orders will be placed until the bot is started."
        )
    elif actually_running and age is not None:
        st.caption(f"Last check-in {_format_age(age)}.")

    if status.last_signal:
        st.caption(f"Last signal: {status.last_signal}")
    if status.last_error:
        st.error(f"Last error: {status.last_error}")

    if kill_switch and st.button("🛑 KILL SWITCH — stop auto-trading now", type="primary", key=kill_key):
        trigger_kill_switch()
        notifications.notify_kill_switch_engaged("the dashboard")
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
    target_ids = [a["id"] for a in linked if a["id"] in _all_target_account_ids(control)]
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


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_news(ticker: str, limit: int) -> list[dict]:
    """Cached ~10 min per (ticker, limit) so flipping to the News page doesn't
    hammer Polygon's rate limit."""
    client = PolygonClient(api_key=keystore.get_key("POLYGON_API_KEY"), use_cache=False)
    return client.get_news(ticker, limit=limit)


def _md_safe(text: str | None) -> str:
    """Escape '$' so Streamlit's markdown doesn't render dollar amounts in news
    prose as LaTeX math (financial headlines are full of '$16 trillion' etc.)."""
    return (text or "").replace("$", "\\$")


def _news_time(published_utc: str | None) -> str:
    if not published_utc:
        return ""
    try:
        dt = datetime.fromisoformat(published_utc.replace("Z", "+00:00"))
        return f"{dt:%b %d, %H:%M} UTC"
    except ValueError:
        return published_utc


def render_news_page() -> None:
    """Read-only news-headlines-per-ticker panel (#21). Nothing here places or
    influences a trade; it's context for the human, and the data source a future
    task will feed into conviction (#25). Uses Polygon's news endpoint."""
    st.subheader("News")
    st.caption(
        "Recent headlines per ticker from Polygon — read-only context. Nothing here places or "
        "influences a trade; it's also the groundwork for feeding news sentiment into conviction later."
    )
    if not keystore.get_key("POLYGON_API_KEY"):
        st.info("No Polygon API key set. Add one in **Settings → API keys** to load news.")
        return

    ticker = st.text_input("Ticker", value="AAPL", key="news_ticker").upper().strip()
    limit = st.slider("Number of headlines", min_value=5, max_value=50, value=10, key="news_limit")
    if not ticker:
        st.info("Enter a ticker to see its latest headlines.")
        return

    try:
        articles = _fetch_news(ticker, limit)
    except PolygonError as e:
        msg = str(e)
        if "NOT_AUTHORIZED" in msg.upper() or "NOT ENTITLED" in msg.upper() or "403" in msg:
            st.warning(
                "Your Polygon plan doesn't appear to include the News endpoint — headlines need a "
                "plan tier with news access."
            )
        else:
            st.error(f"Couldn't load news for {ticker}: {msg}")
        return
    except Exception as e:  # noqa: BLE001 — never traceback the whole page over a news fetch
        st.error(f"Couldn't load news for {ticker}: {e}")
        return

    if not articles:
        st.caption(f"No recent news found for {ticker}.")
        return

    badges = {"positive": "🟢 Positive", "negative": "🔴 Negative", "neutral": "⚪ Neutral"}
    st.caption(f"Showing {len(articles)} recent headlines for {ticker} (cached ~10 min).")
    for a in articles:
        with st.container(border=True):
            title = _md_safe(a.get("title") or "(untitled)")
            url = a.get("article_url")
            st.markdown(f"**[{title}]({url})**" if url else f"**{title}**")
            sentiment = (a.get("sentiment") or "").lower()
            meta = [b for b in [a.get("publisher"), _news_time(a.get("published_utc")), badges.get(sentiment)] if b]
            if meta:
                st.caption(" · ".join(meta))
            if a.get("description"):
                st.write(_md_safe(a["description"]))
            if a.get("sentiment_reasoning"):
                st.caption(f"Why {sentiment}: {_md_safe(a['sentiment_reasoning'])}")


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
            st.Page(render_playlist_page, title="Backtest playlist", icon=":material/queue_music:"),
            st.Page(render_scan_history_tab, title="Scan history", icon=":material/history:"),
            st.Page(render_news_page, title="News", icon=":material/newspaper:"),
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

    st.title("Holotable")
    st.caption("Trading Bot Backtester")
    page.run()


if __name__ == "__main__":
    main()
