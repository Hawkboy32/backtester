"""Roster of eToro investors actively being copied, and the mapping from
their positions to the orders Chopper actually placed on their behalf.
Mirrors roster.py's own RosterEntry/RosterState shape deliberately — same
active/paused concept, same "adjustable roster, not a one-time pick"
philosophy the user asked for (2026-09-16: "keep watching all 5, we can
make decisions... on whether to increase or decrease the roster size").

The open_positions map is the part with no roster.py analogue, and it's the
load-bearing piece: Chopper's own paper-account sizing never matches the
investor's own qty, so a later CLOSE event has no way to know what to sell
without this. Keyed by eToro's own positionId (globally unique across every
investor, confirmed live 2026-09-16/17 watching 5 of them simultaneously) —
NOT by (account, ticker), unlike position_attribution.py's map, because two
different investors (or the same one, twice — confirmed live: Aukie2008
opened/closed the same ticker several times in one day) can hold the same
ticker on the same account at once, and (account, ticker) can't tell those
apart. This is why copy-trade P&L is recorded through live_trades.py
directly (tagged "Copy: <username>", same placeholder-tag spirit as
UNATTRIBUTED_STRATEGY) rather than through position_attribution.py at all.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

COPY_ROSTER_PATH = STATE_DIR / "copy_roster.json"


@dataclass
class CopyRosterEntry:
    username: str
    status: str = "active"  # "active" | "paused"
    added_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    paused_at: str | None = None
    pause_reason: str | None = None


@dataclass
class CopyOrderFill:
    account_id: str
    account_nickname: str
    qty: float
    filled_avg_price: float | None


@dataclass
class OpenCopyPosition:
    username: str
    etoro_position_id: int
    ticker: str
    side: str  # "long" | "short" — the DIRECTION we took, mirrors the investor's
    opened_at: str
    fills: list[CopyOrderFill] = field(default_factory=list)


@dataclass
class CopyRosterState:
    entries: list[CopyRosterEntry] = field(default_factory=list)
    open_positions: list[OpenCopyPosition] = field(default_factory=list)


def _from_dict(cls, data: dict):
    """Same tolerant-load contract as auto_trader_state.from_dict — ignore
    unknown keys, fill in defaults for missing ones, so an older state file
    never hard-fails a load. Nested manually (CopyOrderFill inside
    OpenCopyPosition) since this project's from_dict doesn't recurse."""
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_roster() -> CopyRosterState:
    if not COPY_ROSTER_PATH.exists():
        return CopyRosterState()
    try:
        data = json.loads(COPY_ROSTER_PATH.read_text(encoding="utf-8"))
        entries = [_from_dict(CopyRosterEntry, e) for e in data.get("entries", [])]
        open_positions = []
        for p in data.get("open_positions", []):
            fills = [_from_dict(CopyOrderFill, f) for f in p.get("fills", [])]
            pos = _from_dict(OpenCopyPosition, p)
            pos.fills = fills
            open_positions.append(pos)
        return CopyRosterState(entries=entries, open_positions=open_positions)
    except Exception:
        return CopyRosterState()


def save_roster(state: CopyRosterState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "entries": [asdict(e) for e in state.entries],
        "open_positions": [asdict(p) for p in state.open_positions],
    }
    atomic_write_text(COPY_ROSTER_PATH, json.dumps(data, indent=2))


def active_usernames(state: CopyRosterState) -> list[str]:
    return [e.username for e in state.entries if e.status == "active"]


def ensure_entries(state: CopyRosterState, usernames: list[str]) -> CopyRosterState:
    """Add any username not already tracked, as a new active entry. Never
    touches an existing entry's status — re-running this with the same
    5 names doesn't un-pause one you paused deliberately."""
    known = {e.username for e in state.entries}
    for name in usernames:
        if name not in known:
            state.entries.append(CopyRosterEntry(username=name))
    return state


def find_open_position(state: CopyRosterState, etoro_position_id: int) -> OpenCopyPosition | None:
    return next((p for p in state.open_positions if p.etoro_position_id == etoro_position_id), None)


def remove_open_position(state: CopyRosterState, etoro_position_id: int) -> None:
    state.open_positions = [p for p in state.open_positions if p.etoro_position_id != etoro_position_id]
