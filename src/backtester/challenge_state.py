"""State for a Kraken Funded challenge attempt - a genuinely different kind
of trading problem from everything else in this project, see
challenge_trader.py's own module docstring for why it's a separate process
rather than a special case bolted onto auto_trader.py.

BUILT AHEAD OF CONFIRMATION (2026-09-26): whether Kraken Funded permits
API/bot trading at all is still unconfirmed - not documented anywhere
Kraken publishes, and their own support AI couldn't answer it either (see
RESEARCH conversation same date). This module and challenge_trader.py are
built and ready on the assumption bots ARE allowed, so there's no lost time
if/when that's confirmed - but nothing here should go near a real account
until it actually is confirmed.

TIERS: all three (Starter/Mid/Anchor) share the IDENTICAL 12% profit target
and 3% drawdown floor - confirmed live 2026-09-26, not assumed - only the
starting balance and one-time fee differ. Because position sizing here is a
FRACTION of current equity (not a fixed dollar amount) and both thresholds
are percentages of the starting balance, the pass/fail dynamics are
scale-invariant: whatever sizing works for the $1K tier works identically
for $5K/$10K. Practical upshot - start with the Starter tier ($20 fee) to
prove the approach; there is no edge to paying more per attempt to start.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json

PATH = STATE_DIR / "challenge_state.json"

# All three tiers share identical rules (see module docstring) - only
# starting_balance and fee_usd differ. target/floor are ALWAYS computed as
# +12%/-3% of starting_balance, never hardcoded per tier, so a rule change
# on Kraken's side only needs updating in one place (TARGET_PCT/FLOOR_PCT
# below), not once per tier.
TARGET_PCT = 0.12
FLOOR_PCT = -0.03

TIERS: dict[str, dict] = {
    "starter": {"label": "Starter", "starting_balance": 1_000.0, "fee_usd": 20.0},
    "mid": {"label": "Mid", "starting_balance": 5_000.0, "fee_usd": 50.0},
    "anchor": {"label": "Anchor", "starting_balance": 10_000.0, "fee_usd": 90.0},
}


def target_balance(starting_balance: float) -> float:
    return starting_balance * (1 + TARGET_PCT)


def floor_balance(starting_balance: float) -> float:
    return starting_balance * (1 + FLOOR_PCT)


@dataclass
class ChallengeAttempt:
    """One $-paid attempt at one tier. Terminal the instant status leaves
    'active' - matches the real product ("the challenge ends as soon as
    your balance hits the profit target" / "...drops more than 3%"), not
    a status Chopper decides to keep trading through.
    """
    tier: str
    attempt_number: int
    account_id: str
    started_at: str
    starting_balance: float
    strategy_name: str
    sizing_pct: float  # fraction of current equity per entry - see challenge_trader.py
    status: str = "active"  # "active" | "passed" | "failed"
    ended_at: str | None = None
    ending_balance: float | None = None

    @property
    def target(self) -> float:
        return target_balance(self.starting_balance)

    @property
    def floor(self) -> float:
        return floor_balance(self.starting_balance)

    def progress_fraction(self, current_balance: float) -> float:
        """0.0 at the floor, 1.0 at the target - NOT centered on the
        starting balance, since -3%/+12% is a 15-point range with the
        starting point sitting 20% of the way in from the floor end, not
        in the middle. What the widget's progress bar should actually
        plot, so it doesn't silently misrepresent where zero really is."""
        span = self.target - self.floor
        return (current_balance - self.floor) / span if span else 0.0


@dataclass
class ChallengeHistory:
    attempts: list[ChallengeAttempt] = field(default_factory=list)

    @property
    def active(self) -> ChallengeAttempt | None:
        for a in self.attempts:
            if a.status == "active":
                return a
        return None

    @property
    def next_attempt_number(self) -> int:
        return len(self.attempts) + 1


def load() -> ChallengeHistory:
    raw = read_state_json(PATH, default={"attempts": []})
    return ChallengeHistory(attempts=[ChallengeAttempt(**a) for a in raw.get("attempts", [])])


def save(history: ChallengeHistory) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, json.dumps({"attempts": [asdict(a) for a in history.attempts]}, indent=2))


def start_attempt(tier: str, account_id: str, strategy_name: str, sizing_pct: float) -> ChallengeAttempt:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier {tier!r} - must be one of {list(TIERS)}")
    history = load()
    if history.active is not None:
        raise ValueError(
            f"Attempt #{history.active.attempt_number} ({history.active.tier}) is still active - "
            "end it (pass/fail) before starting a new one. Only one challenge attempt runs at a time."
        )
    attempt = ChallengeAttempt(
        tier=tier,
        attempt_number=history.next_attempt_number,
        account_id=account_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        starting_balance=TIERS[tier]["starting_balance"],
        strategy_name=strategy_name,
        sizing_pct=sizing_pct,
    )
    history.attempts.append(attempt)
    save(history)
    return attempt


def end_attempt(status: str, ending_balance: float) -> ChallengeAttempt:
    """status: 'passed' or 'failed'. Raises if no attempt is active - ending
    one is only ever meaningful once, never a no-op retry."""
    if status not in ("passed", "failed"):
        raise ValueError(f"status must be 'passed' or 'failed', got {status!r}")
    history = load()
    attempt = history.active
    if attempt is None:
        raise ValueError("No active challenge attempt to end.")
    attempt.status = status
    attempt.ended_at = datetime.now(timezone.utc).isoformat()
    attempt.ending_balance = ending_balance
    save(history)
    return attempt
