"""Shared snapshot of the auto-trader's most recently computed signal for
every ticker it evaluates, keyed by ticker.

Exists so a SECOND process (the Mobile_App backend) doesn't need its own
independent Polygon fetch just to show "what would the strategy say right
now" for the exact same tickers the live bot already evaluates every cycle.
Both processes sharing one Polygon account and each self-throttling to their
OWN 5-req/min budget doesn't prevent real 429s when their cycles land close
together — the fix is not fetching the same data twice, not raising the
limit (the free tier really is ~5/min).

Written by auto_trader.py's _trade_target (which already has bars/signal/
conviction in hand, so this costs zero extra Polygon calls), read by
Mobile_App/backend/signal_service.py. A reader should treat an entry as
stale (and fall back to its own direct fetch) once it's older than a few
poll intervals — auto_trader.py isn't guaranteed to be running.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

SIGNALS_PATH = STATE_DIR / "current_signals.json"


def record_signal(
    ticker: str,
    strategy_name: str,
    signal: str,
    price: float,
    conviction: float | None,
    bar_timestamp: str,
) -> None:
    """Read-modify-write one ticker's entry. Never raises - a snapshot write
    failing must never be allowed to interrupt the actual trading cycle."""
    try:
        data = load_signals()
        data[ticker] = {
            "strategy_name": strategy_name,
            "signal": signal,
            "price": price,
            "conviction": conviction,
            "bar_timestamp": bar_timestamp,
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_text(SIGNALS_PATH, json.dumps(data, indent=2))
    except Exception:  # noqa: BLE001
        pass


def load_signals() -> dict[str, dict]:
    if not SIGNALS_PATH.exists():
        return {}
    try:
        return json.loads(SIGNALS_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
