"""Dead-man's switch: an OUTBOUND heartbeat to an external monitor that alerts
the user when the bot stops checking in.

WHY THIS IS NOT A LOCAL WATCHDOG. On 2026-07-22 the auto-trader died when the
laptop slept and stayed dead ~29 hours with an open position — and nothing
alerted, because silence looks exactly like "no trades today". A watchdog
process on the same laptop cannot fix that: whatever kills or freezes the bot
(sleep, shutdown, power cut, crash) freezes the watchdog too. The only thing
that can notice absence is something running somewhere else, so the alert has
to be inverted — the bot proves it is alive on a schedule, and an external
service pages the user when the proof stops arriving.

Provider: healthchecks.io (or any self-hosted instance / compatible service) —
a plain HTTPS GET to a per-check URL is the entire protocol, so there is no SDK
and no API key. The check's expected schedule and grace period are configured on
that side, NOT here; see the README note in Settings. That matters for the
user's routine of shutting the laptop down overnight: set the check's cron
schedule to the hours the bot is meant to be up (US market hours) so an
expected overnight silence does not page anyone. Alerts you learn to ignore are
worse than no alerts.

SAFETY: ping() never raises and never blocks for long. The trade is sacred; the
heartbeat is a nicety. Same contract as notifications.notify().

SECURITY NOTE: the ping URL contains a UUID that is mildly sensitive — anyone
holding it could send fake "I'm alive" pings and suppress a real alert. It
cannot place a trade, move money, or read anything, so it lives in the
gitignored state file next to the other local config rather than in the keyring
with the broker credentials.
"""

from __future__ import annotations

import ipaddress
import json
import time
from urllib.parse import urlparse

import requests

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

CONFIG_PATH = STATE_DIR / "heartbeat.json"
PING_TIMEOUT_SECONDS = 5
# Floor between pings, so a short poll interval can't hammer the monitor. A
# dead-man's switch is judged in minutes, so one ping a minute is plenty.
MIN_PING_INTERVAL_SECONDS = 60

_last_ping_monotonic: float | None = None


def _default_config() -> dict:
    return {"enabled": False, "ping_url": ""}


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return _default_config()
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_config()


def save_config(cfg: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # atomic_write_text, not write_text: the dashboard writes this file while the
    # standalone trader reads it (see auto_trader_state.atomic_write_text).
    atomic_write_text(CONFIG_PATH, json.dumps(cfg, indent=2))


def is_valid_ping_url(url: str) -> bool:
    """Cheap sanity check so a pasted-wrong value fails in Settings, where the
    user can see it, rather than silently never alerting.

    HTTPS is required for anything on the public internet — the ping URL is a
    bearer token in a query path, and sending it in clear text across the
    network would let anyone in the middle suppress the alert. Plain HTTP is
    allowed ONLY for loopback and private LAN addresses, so a self-hosted
    monitor (the Raspberry Pi plan) works without a certificate.
    """
    url = (url or "").strip()
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if not parsed.path.strip("/"):
        return False  # a bare host is never a ping URL; the check id lives in the path
    if parsed.scheme == "https":
        return True
    return _is_private_host(parsed.hostname)


def _is_private_host(hostname: str) -> bool:
    """True for loopback / private-range / .local hosts, i.e. somewhere the
    traffic never leaves the user's own network."""
    if hostname == "localhost" or hostname.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def ping(suffix: str = "", config: dict | None = None, force: bool = False) -> bool:
    """Tell the monitor the bot is alive. Returns True if the ping was delivered.

    Never raises: a dead network, a typo'd URL, a monitor outage — none of it can
    reach the trading loop. Returns False for "not sent" of any kind, including a
    throttled call, because no caller should branch on the difference.

    `suffix` appends a healthchecks.io action ("start", "fail", "log"). `force`
    bypasses the throttle for one-off pings like the Settings test button.
    """
    global _last_ping_monotonic
    try:
        cfg = config if config is not None else load_config()
        if not cfg.get("enabled"):
            return False
        url = (cfg.get("ping_url") or "").strip()
        if not is_valid_ping_url(url):
            return False

        now = time.monotonic()
        if (
            not force
            and _last_ping_monotonic is not None
            and now - _last_ping_monotonic < MIN_PING_INTERVAL_SECONDS
        ):
            return False

        if suffix:
            url = f"{url.rstrip('/')}/{suffix}"
        resp = requests.get(url, timeout=PING_TIMEOUT_SECONDS)
        resp.raise_for_status()
        # Only a DELIVERED ping resets the throttle, so a monitor outage doesn't
        # silence us for a minute after each failure.
        _last_ping_monotonic = now
        return True
    except Exception:
        return False
