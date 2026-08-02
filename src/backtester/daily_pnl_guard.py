"""Daily P&L giveback guard.

Distinct from the two risk controls that already exist:
- `max_trades_per_day` (auto_trader_state.py) is a trade-COUNT cap.
- account_risk.py's breaker is a hard, ACCOUNT-LIFETIME, high-water-mark
  drawdown stop that requires a deliberate manual re-arm once tripped.

This one is lighter and resets every day on its own: once an account's
profit for TODAY has retraced more than `giveback_pct` off today's own
intraday peak profit, new entries are blocked for the rest of the day —
existing positions can still be closed. Source idea (see CLAUDE_NOTES.txt
"PENDING IDEAS", added 2026-07-31): "cap red days, don't cap green days" —
protects gains already made on a hot day without capping the upside of a
day that keeps running, and without needing an outright loss to trigger
(unlike the account-level breaker, which only fires on real drawdown from
the all-time peak, not a giveback of today's own gains specifically).

Only evaluated once today's peak P&L is positive — a day that never got
into profit has nothing to give back, and the account-level breaker (or a
straightforward loss limit) is the right tool for that case, not this one.

No manual re-arm exists here on purpose: unlike account_risk.py's lifetime
breach, this is meant to clear itself automatically at the start of the
next trading day, not require a human to notice and reset it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

PATH = STATE_DIR / "daily_pnl_guard.json"


def load_state() -> dict:
    if not PATH.exists():
        return {}
    try:
        return json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, json.dumps(state, indent=2))


def _today_str() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def check_and_update(account_id: str, current_equity: float, giveback_pct: float) -> tuple[bool, str | None]:
    """Call once per account per cycle. Returns (blocked, reason).

    Resets automatically (fresh starting_equity, peak_pnl, unblocked) the
    first time this is called on a new UTC date for this account.
    """
    state = load_state()
    today = _today_str()
    entry = state.get(account_id)

    if entry is None or entry.get("date") != today:
        entry = {
            "date": today,
            "starting_equity": current_equity,
            "peak_pnl": 0.0,
            "blocked": False,
            "reason": None,
        }

    if not entry["blocked"]:
        current_pnl = current_equity - entry["starting_equity"]
        entry["peak_pnl"] = max(entry["peak_pnl"], current_pnl)
        if entry["peak_pnl"] > 0:
            floor = entry["peak_pnl"] * (1 - giveback_pct / 100)
            if current_pnl <= floor:
                entry["blocked"] = True
                entry["reason"] = (
                    f"today's P&L ${current_pnl:,.2f} has given back more than {giveback_pct:.0f}% "
                    f"of today's peak profit (${entry['peak_pnl']:,.2f})"
                )

    state[account_id] = entry
    save_state(state)
    return entry["blocked"], entry["reason"]


def get_status(account_id: str) -> dict | None:
    entry = load_state().get(account_id)
    if entry is None or entry.get("date") != _today_str():
        return None  # stale (yesterday's) or never-seen — nothing live to report
    return entry
