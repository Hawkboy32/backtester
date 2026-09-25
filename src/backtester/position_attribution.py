"""Tracks which strategy is currently responsible for each open live/paper
position, since the broker itself has no concept of "strategy" — a position
is just a ticker + qty on an account.

Not safety-critical the way auto_trader_state's control file is: if this
file is missing or corrupted, the worst outcome is a live trade going
unattributed (skipped for live_trades.py logging, still executed and logged
to execution_log.py normally) — never a reason to block or misrepresent an
actual trade. So unlike auto_trader_state.load_control's fail-closed
("treat as killed"), a bad file here just resets to "nothing attributed".
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json

PATH = STATE_DIR / "positions.json"


def _key(account_id: str, ticker: str) -> str:
    return f"{account_id}|{ticker}"


def load_map() -> dict[str, dict]:
    """Raises StateFileUnreadable rather than returning {} on a read failure —
    every caller below is a read-modify-write, so a silent empty return would
    let the next save wipe every open position's strategy attribution, and a
    close with no attribution is never recorded to live_trades (its P&L just
    disappears). See read_state_json for the incident this pattern caused."""
    return read_state_json(PATH, default={})


def save_map(data: dict[str, dict]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, json.dumps(data, indent=2))


def record_open(
    account_id: str, ticker: str, strategy_name: str, conviction: float | None = None,
    sizing_mode: str | None = None, sizing_value: float | None = None,
    dollars_committed: float | None = None, entry_price: float | None = None,
) -> None:
    """sizing_mode/sizing_value/dollars_committed: the EFFECTIVE sizing this
    specific entry actually used (2026-08-16) - already resolved through any
    per-account slide/override and the GARCH size_multiplier, i.e. exactly
    what was live at the moment of this fill, not the global control.json
    setting (which may since have changed). All three None (the default) for
    a caller that doesn't have this context, e.g. an old call site or a
    position opened before this existed - the mobile /positions endpoint
    treats that as "sizing unknown" rather than guessing.

    entry_price (2026-09-22): the REAL fill price from the opening order's
    own OrderResult, captured here because it's the only reliable source —
    some brokers never populate a position's avg_entry_price at all. Found
    live on KRKAPI: Kraken's Balance/TradeBalance endpoints have no
    cost-basis field, so kraken.py's get_positions() deliberately always
    reports avg_entry_price=0.0 (see that module's docstring) - but the
    CLOSING code here used to trust that broker-reported 0.0 anyway,
    recording entry_price=0 and inflating pnl to nearly the full exit
    notional. 14 of 15 KRKAPI round trips were wrong this way, overstating
    live P&L by ~$109 against a true ~$0.01. Recording the real fill price
    at open time sidesteps the broker's limitation entirely."""
    data = load_map()
    data[_key(account_id, ticker)] = {
        "strategy_name": strategy_name,
        "opened_at": datetime.now(timezone.utc).isoformat(),
        "conviction": conviction,  # [0,1] entry-signal strength (#25); carried through to live_trades on close
        "sizing_mode": sizing_mode,
        "sizing_value": sizing_value,
        "dollars_committed": dollars_committed,
        "entry_price": entry_price,
    }
    save_map(data)


def pop_open(account_id: str, ticker: str) -> dict | None:
    data = load_map()
    entry = data.pop(_key(account_id, ticker), None)
    if entry is not None:
        save_map(data)
    return entry


def resolve_entry_price(attribution: dict, broker_reported_entry_price: float) -> float:
    """The entry price to actually RECORD for a closed round trip.

    Prefer the real fill price captured at open time (attribution's own
    "entry_price", added 2026-09-22) over whatever the broker's position
    object reports now — some brokers (Kraken confirmed) never populate
    avg_entry_price at all, always reporting 0.0 (no cost-basis field in
    their API), and trusting that blindly wrecked recorded P&L: 14 of 15
    KRKAPI round trips got entry_price=0, overstating live P&L by ~$109
    against a true ~$0.01. Falls back to the broker's value only when no
    recorded entry_price exists (a position opened before this fix, or
    whose attribution was lost/never captured)."""
    return attribution.get("entry_price") or broker_reported_entry_price


def reconcile(account_id: str, actually_held_tickers: set[str]) -> None:
    """Drop any attribution entries for this account whose ticker isn't
    actually held anymore (e.g. closed manually, outside auto_trader.py's own
    BUY/SELL flow). Call once per poll cycle, before evaluating signals, using
    a get_positions() result you're already fetching — bounds staleness to
    one poll interval at no extra broker calls.
    """
    data = load_map()
    prefix = f"{account_id}|"
    stale = [
        key for key in data
        if key.startswith(prefix) and key[len(prefix):] not in actually_held_tickers
    ]
    if not stale:
        return
    for key in stale:
        del data[key]
    save_map(data)
