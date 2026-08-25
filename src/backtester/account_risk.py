"""Account-level max-drawdown circuit breaker.

Tracks each linked account's peak equity since it started being watched, and
flags an account as "risk-blocked" once its current equity has fallen more
than a configured percentage below that peak. A blocked account is a hard
stop: no new entries (BUY) are allowed for it in either manual or roster
mode, though it can still close existing positions — de-risking is always
allowed, adding risk never is once blocked. Blocking persists across poll
cycles and requires a deliberate, manual re-arm (not an automatic recovery
once equity climbs back up) — breaching a real risk limit is meant to get a
human's attention, not quietly self-heal.

Re-arming resets the peak to the account's equity at the moment of re-arm,
not the old (higher) peak — otherwise the very next check would immediately
re-trip the same breach again, making the re-arm button pointless.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json

PATH = STATE_DIR / "account_risk.json"


def load_state() -> dict:
    """Raises StateFileUnreadable rather than returning {} on a read failure.
    check_and_update below is a read-modify-write, so a silent empty return
    would reset every account's peak_equity AND forget blocked=True — an
    account halted by the drawdown circuit breaker would quietly un-block
    itself. Failing loud keeps the breaker latched. See read_state_json."""
    return read_state_json(PATH, default={})


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, json.dumps(state, indent=2))


def check_and_update(account_id: str, current_equity: float, max_drawdown_pct: float) -> tuple[bool, str | None]:
    """Call once per account per cycle. Returns (blocked, reason).

    Updates the account's peak equity and, if not already blocked, checks
    for a new breach. Once blocked, stays blocked (same reason) regardless
    of how equity moves afterward, until reset_breach() is called.
    """
    state = load_state()
    entry = state.get(
        account_id,
        {"peak_equity": current_equity, "blocked": False, "reason": None, "blocked_at": None},
    )

    if not entry["blocked"]:
        entry["peak_equity"] = max(entry["peak_equity"], current_equity)
        floor = entry["peak_equity"] * (1 - max_drawdown_pct / 100)
        if current_equity < floor:
            entry["blocked"] = True
            entry["reason"] = (
                f"equity ${current_equity:,.2f} is more than {max_drawdown_pct:.1f}% below "
                f"peak ${entry['peak_equity']:,.2f}"
            )
            entry["blocked_at"] = datetime.now(timezone.utc).isoformat()

    state[account_id] = entry
    save_state(state)
    return entry["blocked"], entry["reason"]


def reset_breach(account_id: str, current_equity: float) -> None:
    """Manual re-arm: clears the block and resets the peak to the current
    equity, so risk is measured fresh from this point forward.
    """
    state = load_state()
    state[account_id] = {
        "peak_equity": current_equity,
        "blocked": False,
        "reason": None,
        "blocked_at": None,
    }
    save_state(state)


def get_status(account_id: str) -> dict | None:
    return load_state().get(account_id)
