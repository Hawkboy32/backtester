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

from backtester.auto_trader_state import STATE_DIR

CONFIG_PATH = STATE_DIR / "notifications.json"
NTFY_BASE = "https://ntfy.sh"
SEND_TIMEOUT_SECONDS = 5


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {"enabled": False, "provider": "ntfy", "ntfy_topic": ""}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"enabled": False, "provider": "ntfy", "ntfy_topic": ""}


def save_config(cfg: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _send_ntfy(topic: str, title: str, message: str, priority: str | None = None) -> bool:
    headers = {"Title": title}
    if priority:
        headers["Priority"] = priority
    resp = requests.post(
        f"{NTFY_BASE}/{topic}",
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
            return _send_ntfy(topic, title, message, priority)
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
