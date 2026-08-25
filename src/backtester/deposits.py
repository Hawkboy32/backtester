"""Manual deposit ledger per linked account - lets the user record when they
put real money into an account, so the dashboard can show a "true P&L"
(current equity minus money actually deposited) that isn't inflated by the
deposit itself, separate from the broker's own realized/unrealized trading
P&L numbers.

Purely a manual log - no broker here exposes transfer/deposit history
simply or uniformly enough to read this automatically, so it's only as
accurate as what actually gets recorded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json

DEPOSITS_PATH = STATE_DIR / "deposits.json"


@dataclass
class Deposit:
    amount: float
    date: str  # YYYY-MM-DD the deposit was actually made, user-supplied
    note: str = ""
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def load_all() -> dict[str, list[Deposit]]:
    """Raises StateFileUnreadable if the file exists but can't be parsed —
    NEVER returns {} on failure. This log is the only record of money paid in
    (no broker exposes transfer history), and every writer below is a
    read-modify-write, so a silent empty return here would let the next
    recorded deposit erase the entire history. Demonstrated live 2026-08-12:
    $53.76 of real deposits vanished in a test when an unknown field made the
    old `except Exception: return {}` fire. Unknown keys are ignored rather
    than fatal so a NEWER writer's extra field can't lock an older reader out
    (the exact shape that destroyed roster.json the same day)."""
    data = read_state_json(DEPOSITS_PATH, default={})
    known = {f.name for f in fields(Deposit)}
    return {
        account_id: [Deposit(**{k: v for k, v in d.items() if k in known}) for d in entries]
        for account_id, entries in data.items()
    }


def _save_all(data: dict[str, list[Deposit]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {account_id: [asdict(d) for d in entries] for account_id, entries in data.items()}
    atomic_write_text(DEPOSITS_PATH, json.dumps(payload, indent=2))


def record_deposit(account_id: str, amount: float, deposit_date: str, note: str = "") -> None:
    if amount <= 0:
        raise ValueError("Deposit amount must be positive.")
    data = load_all()
    data.setdefault(account_id, []).append(Deposit(amount=amount, date=deposit_date, note=note))
    _save_all(data)


def remove_deposit(account_id: str, index: int) -> None:
    """index into deposits_for(account_id)'s own list, most-recent-first (see
    deposits_for) - callers should always re-fetch that same ordering right
    before showing a remove control, never cache an index across a rerun."""
    data = load_all()
    entries = data.get(account_id, [])
    forward_index = len(entries) - 1 - index
    if 0 <= forward_index < len(entries):
        entries.pop(forward_index)
        _save_all(data)


def total_deposited(account_id: str) -> float:
    data = load_all()
    return sum(d.amount for d in data.get(account_id, []))


def deposits_for(account_id: str) -> list[Deposit]:
    """Most-recent-first (by recorded_at) - matches how every other history
    list in this project displays (roster events, trade history)."""
    data = load_all()
    return sorted(data.get(account_id, []), key=lambda d: d.recorded_at, reverse=True)
