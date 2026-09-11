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
    # What this deposit's currency ACTUALLY turned into once really converted
    # (e.g. GBP -> USD on the exchange), vs `amount` above which is whatever
    # the account's live equity read showed at logging time - for a
    # foreign-currency deposit sitting unconverted, that's a real, honest
    # snapshot, but it's a live FX-rate estimate, not the amount that lands
    # once conversion actually executes days later, and it can't know about
    # a conversion fee that hasn't happened yet either. Both numbers are
    # correct answers to different questions (found live 2026-09-06: a
    # GBP deposit's `amount` was logged as $16.22, the real conversion later
    # banked $15.81 - neither was wrong, they're not the same question).
    # None until record_conversion() below fills it in.
    converted_amount: float | None = None
    converted_at: str | None = None  # ISO timestamp of the real conversion, when known


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


def record_conversion(account_id: str, converted_total: float, converted_at: str | None = None) -> None:
    """Attach a REAL currency-conversion result to whichever deposits for
    this account are still awaiting one (converted_amount is None) - the
    realistic pattern (confirmed live 2026-09-06): several small foreign-
    currency deposits sit unconverted, then get bulk-converted together in
    one exchange transaction, so there's no clean 1:1 deposit-to-conversion
    mapping to begin with. Splits converted_total across those entries
    proportionally to their own logged `amount` (each deposit's live-equity
    estimate at the time), which is the best available signal for how much
    of the real total belongs to each one - not exact, but far better than
    attributing it all to the single triggering entry.

    Raises ValueError if there's nothing unconverted to attach this to,
    rather than silently doing nothing - a conversion that can't be matched
    to a real pending deposit is worth surfacing, not swallowing.
    """
    data = load_all()
    entries = data.get(account_id, [])
    pending = [d for d in entries if d.converted_amount is None]
    if not pending:
        raise ValueError(f"No unconverted deposits recorded for account {account_id!r} to attach this to.")
    pending_total = sum(d.amount for d in pending)
    at = converted_at or datetime.now(timezone.utc).isoformat()
    for d in pending:
        share = (d.amount / pending_total) if pending_total else 0.0
        d.converted_amount = round(converted_total * share, 2)
        d.converted_at = at
    _save_all(data)


def total_deposited(account_id: str) -> float:
    """The best-known real total: a deposit's actual converted_amount once
    known, falling back to its live-equity-estimate `amount` for anything
    still sitting unconverted. Automatically gets more accurate as
    record_conversion() fills in real numbers, without needing every caller
    to know which figure is which - see Deposit's own docstring."""
    data = load_all()
    return sum((d.converted_amount if d.converted_amount is not None else d.amount) for d in data.get(account_id, []))


def total_deposited_estimated(account_id: str) -> float:
    """The ORIGINAL logged total, ignoring any later real conversion -
    kept separately so a trend (is the live-equity estimate consistently
    over/understating what actually lands?) stays visible instead of being
    silently overwritten once record_conversion() runs."""
    data = load_all()
    return sum(d.amount for d in data.get(account_id, []))


def deposits_for(account_id: str) -> list[Deposit]:
    """Most-recent-first (by recorded_at) - matches how every other history
    list in this project displays (roster events, trade history)."""
    data = load_all()
    return sorted(data.get(account_id, []), key=lambda d: d.recorded_at, reverse=True)
