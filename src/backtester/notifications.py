"""Push notifications for trade events (open/close), sent to the user's phone.

Provider-agnostic by design; ntfy (https://ntfy.sh) is the first and default
backend — zero-setup (install the app, subscribe to a topic, done), no secret
to manage, and self-hostable on a home server later. A Telegram backend can be
added alongside without touching callers.

Config lives in a JSON file in auto_trader_state/ (like control.json) so both
the dashboard (which writes it) and the standalone auto_trader process (which
reads it) see the same settings — file-based coordination, matching the rest
of this project.

SAFETY: notify() never raises. A dead network, a bad topic, a timeout — none of
it can propagate to the caller. A trade must never be blocked, delayed, or
aborted because a notification failed. The trade is sacred; the ping is a nicety.
"""

from __future__ import annotations

import json

import requests

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

CONFIG_PATH = STATE_DIR / "notifications.json"
DEFAULT_NTFY_BASE = "https://ntfy.sh"
SEND_TIMEOUT_SECONDS = 5

_DEFAULT_CONFIG = {"enabled": False, "provider": "ntfy", "ntfy_topic": "", "ntfy_base": DEFAULT_NTFY_BASE}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return dict(_DEFAULT_CONFIG)
    try:
        # merge over the defaults so an older config file (saved before
        # ntfy_base existed) still gets the public server rather than a
        # missing/blank base URL.
        return {**_DEFAULT_CONFIG, **json.loads(CONFIG_PATH.read_text(encoding="utf-8"))}
    except Exception:
        return dict(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(CONFIG_PATH, json.dumps(cfg, indent=2))


def _send_ntfy(topic: str, title: str, message: str, priority: str | None = None, base_url: str = DEFAULT_NTFY_BASE) -> bool:
    headers = {"Title": title}
    if priority:
        headers["Priority"] = priority
    resp = requests.post(
        f"{base_url.rstrip('/')}/{topic}",
        data=message.encode("utf-8"),
        headers=headers,
        timeout=SEND_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return True


def notify(title: str, message: str, priority: str | None = None, config: dict | None = None) -> bool:
    """Fire a notification. Returns True on success, False on any failure —
    but NEVER raises, so it is safe to call directly from the trading loop.
    Pass an explicit `config` to bypass the file read (e.g. the dashboard's
    'send test' button testing unsaved settings)."""
    try:
        cfg = config if config is not None else load_config()
        if not cfg.get("enabled"):
            return False
        provider = cfg.get("provider", "ntfy")
        if provider == "ntfy":
            topic = (cfg.get("ntfy_topic") or "").strip()
            if not topic:
                return False
            base_url = (cfg.get("ntfy_base") or DEFAULT_NTFY_BASE).strip() or DEFAULT_NTFY_BASE
            return _send_ntfy(topic, title, message, priority, base_url)
        return False
    except Exception:
        # Swallow everything — a failed ping must never disturb trading.
        return False


def notify_trade_open(ticker: str, strategy_name: str, qty: float, nickname: str, is_paper: bool) -> bool:
    mode = "paper" if is_paper else "LIVE"
    return notify(
        f"Bought {ticker}",
        f"{strategy_name} opened {qty} {ticker} on {nickname} ({mode}).",
    )


def notify_trade_close(ticker: str, strategy_name: str, qty: float, pnl: float, nickname: str, is_paper: bool) -> bool:
    mode = "paper" if is_paper else "LIVE"
    amount = f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}"  # -$8.50, not $-8.50
    return notify(
        f"Sold {ticker} ({amount})",
        f"{strategy_name} closed {qty} {ticker} on {nickname} ({mode}). Realized P&L {amount}.",
    )


def notify_order_rejected(ticker: str, side: str, nickname: str, error: str) -> bool:
    """Previously a silent gap — a rejected order (bad size, insufficient
    funds, broker validation error, etc.) updated status.last_error for the
    dashboard to show, but never pinged the phone the way a successful
    open/close does. High priority: an order that should have gone through
    but didn't is exactly the kind of thing worth an active alert for, not
    just a log line someone might check later."""
    return notify(
        f"Order rejected: {ticker}",
        f"{side.upper()} on {nickname} failed: {error}",
        priority="high",
    )


def notify_drawdown_blocked(nickname: str, reason: str) -> bool:
    """Fired once, at the moment the max-drawdown circuit breaker TRIPS
    (caller is responsible for only calling this on the blocked/not-blocked
    transition, not every cycle it stays blocked — see auto_trader.py's
    run_cycle) — new entries are blocked on this account until a manual
    re-arm, so this is worth surfacing actively, not just in the dashboard."""
    return notify(
        f"Risk limit breached: {nickname}",
        f"New entries blocked — {reason}. Re-arm from the dashboard's Settings page.",
        priority="urgent",
    )


def notify_giveback_blocked(nickname: str, reason: str) -> bool:
    """Same transition-only-fire contract as notify_drawdown_blocked, for
    the lighter daily P&L giveback guard — resets itself next trading day,
    so this is informational rather than needing a re-arm."""
    return notify(
        f"Daily giveback limit reached: {nickname}",
        f"New entries blocked for the rest of today — {reason}.",
        priority="high",
    )


def notify_kill_switch_engaged(source: str) -> bool:
    """Fired at the point trigger_kill_switch() is actually called (dashboard
    button or mobile /kill endpoint) — not from inside auto_trader.py's poll
    loop, so this fires exactly once per real kill event, never repeated
    while the bot stays killed."""
    return notify(
        "Trading stopped",
        f"Kill switch engaged from {source}. Bot will not open or manage new trades until re-armed.",
        priority="urgent",
    )


def notify_manual_trade_suggestion(ticker: str, side: str, qty: float, price: float, tier: str) -> bool:
    """Kraken Funded (the Prop/challenge product) has no API at all — confirmed
    both live (the regular Kraken API credentials see only the normal spot
    account, nothing resembling a challenge balance) and from Kraken's own
    support docs ("Kraken Prop does not currently offer API access... all
    orders are placed manually"). This is the notification half of trading
    a challenge attempt BY HAND with Chopper's help: it still computes the
    signal, still sizes the trade — it just can't press the button, so it
    tells the user exactly what to do instead. See challenge_notifier.py."""
    action = "BUY" if side == "buy" else "CLOSE / SELL"
    return notify(
        f"Kraken Funded: {action} {ticker}",
        f"{action} ~{qty:.6f} {ticker} at ~${price:,.2f} — Kraken Funded has no API, place this "
        f"yourself in the Kraken app. ({tier.title()} tier attempt)",
        priority="high",
    )


def notify_manual_trade_result(ticker: str, pnl: float, pnl_pct: float) -> bool:
    amount = f"{'+' if pnl >= 0 else '-'}${abs(pnl):,.2f}"
    return notify(
        f"Kraken Funded closed ({amount})",
        f"Estimated P&L on that {ticker} round-trip: {amount} ({pnl_pct:+.1f}%). Estimate only, based "
        f"on the suggested prices — confirm your real balance in the Kraken app.",
    )


def notify_manual_challenge_ended(status: str, tier: str, estimated_balance: float, target_or_floor: float) -> bool:
    verb = "PASSED" if status == "passed" else "FAILED"
    boundary = "target" if status == "passed" else "floor"
    return notify(
        f"Kraken Funded challenge likely {verb} ({tier})",
        f"Estimated balance ${estimated_balance:,.2f} crossed the {boundary} (${target_or_floor:,.2f}). "
        f"This is an ESTIMATE based on the suggested trades, not your real account — confirm the actual "
        f"balance in the Kraken app before treating this attempt as officially over.",
        priority="high",
    )


def notify_roster_recommendation_ready(summary: list[str]) -> bool:
    """Fired once when auto_trader.py's once-daily post-close roster check
    computes a real (non-empty) recommendation — see roster.compute_recommendation.
    Nothing has actually changed yet; this just tells the user a change is
    ready to review and approve/dismiss in the mobile app or dashboard."""
    return notify(
        "Roster change ready to review",
        "Nothing applied yet — approve or dismiss in the app.\n" + "\n".join(summary),
        priority="default",
    )
