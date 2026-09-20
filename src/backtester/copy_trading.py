"""eToro copy-trading signal source (2026-09-16). Personal system only — NOT
part of ChopperCommercial (see that repo's own README: it deliberately ships
without this).

Poll a chosen eToro Popular Investor's live portfolio (via GET .../people/
{username}/portfolio/live, the same real endpoint EToroBroker's module
docstring points at), diff it against what was last seen, and turn each
position open/close into a CopyEvent — the same BUY/SELL vocabulary every
strategy already speaks, so this can eventually feed the roster/execution
pipeline the same way. For now this module only OBSERVES and PERSISTS state;
nothing here places an order. See copy_trader.py for the runnable loop that
uses it.

Diffing is by eToro's own stable positionId, not by (instrument, direction)
heuristics — an investor's live-portfolio response carries a real, stable
per-position ID, so "is this the same position as last poll" is an exact
set-membership check, not a guess.

Instrument IDs resolve to real tickers via the market-data/instruments
lookup — confirmed live 2026-09-16 against a real Popular Investor
(instrument 1002/1003/1005/1035 -> GOOG/META/AMZN/WMT). Equity and forex
instruments map onto tickers Chopper's OWN broker accounts (Alpaca, IBKR,
OANDA, IG) already understand directly — a copied position doesn't have to
execute through eToro itself. Anything else (commodities, indices, CFDs,
crypto in a shape not verified yet) resolves to ticker=None: visible in the
event log, but not auto-routable anywhere yet, rather than guessing a wrong
mapping silently.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

PORTFOLIO_URL = "https://public-api.etoro.com/api/v1/user-info/people/{username}/portfolio/live"
INSTRUMENTS_URL = "https://public-api.etoro.com/api/v2/market-data/instruments"

COPY_STATE_DIR = STATE_DIR / "copy_trading"


class CopyTradingError(RuntimeError):
    pass


def _headers(api_key: str, user_key: str) -> dict:
    return {
        "x-request-id": str(uuid.uuid4()),
        "x-api-key": api_key,
        "x-user-key": user_key,
        "Content-Type": "application/json",
    }


def _to_chopper_ticker(symbol: str | None, instrument_type: str | None) -> str | None:
    """Best-effort mapping from an eToro (symbol, type) pair to the ticker
    convention Chopper's own strategies/brokers already use (universe.py's
    crypto/forex CSVs, Polygon convention) — plain for equities, "C:" for
    forex, "X:" for crypto. Returns None where the mapping isn't verified
    (commodities, indices, CFDs, or a crypto symbol shape not confirmed live
    yet) rather than guessing.

    "ETF" confirmed live 2026-09-17 (instrument 3025 -> symbol "GLD", eToro
    type "ETF") — an ordinary NYSE Arca-listed ticker, tradeable through
    Alpaca/IBKR exactly like any equity, same as "Stocks". "Indices"
    confirmed live the same day (instrument 32 -> "GER40", a DAX index CFD)
    — deliberately left unmapped: none of Chopper's brokers trade index
    CFDs, so returning a ticker here would claim a routing this app can't
    actually honour.
    """
    if not symbol or not instrument_type:
        return None
    if instrument_type in ("Stocks", "ETF"):
        return symbol
    if instrument_type == "Forex":
        return f"C:{symbol}"
    if instrument_type == "Crypto" and symbol.endswith("USD"):
        return f"X:{symbol}"
    return None


@dataclass
class WatchedPosition:
    position_id: int
    instrument_id: int
    symbol: str | None
    instrument_type: str | None
    is_buy: bool
    investment_pct: float
    open_timestamp: str
    open_rate: float = 0.0  # the investor's own entry price — used as a sizing
    # REFERENCE only (compute_qty_for_account's own contract: converts a
    # %-equity/fixed-dollar target into a share count, never used for the
    # actual fill), not copied as if it were a live quote.

    @property
    def ticker(self) -> str | None:
        return _to_chopper_ticker(self.symbol, self.instrument_type)


@dataclass
class CopyEvent:
    action: str  # "opened" | "closed"
    username: str
    detected_at: str
    position: WatchedPosition

    def describe(self) -> str:
        if self.position.ticker:
            t = self.position.ticker
        elif self.position.symbol:
            # Resolved to a real eToro instrument but no Chopper-side broker
            # trades that asset class (e.g. an index CFD) — show what it
            # actually is rather than just an opaque numeric ID.
            t = f"{self.position.symbol} ({self.position.instrument_type}, eToro-only — no broker route)"
        else:
            t = f"eToro instrument {self.position.instrument_id} (unresolved)"
        direction = "long" if self.position.is_buy else "short"
        verb = "OPENED" if self.action == "opened" else "CLOSED"
        return f"{self.username} {verb} {direction} {t} (positionId={self.position.position_id})"


def fetch_investor_positions(username: str, api_key: str, user_key: str) -> list[dict]:
    """Raw position rows from eToro's live-portfolio endpoint for one investor."""
    resp = requests.get(
        PORTFOLIO_URL.format(username=username), headers=_headers(api_key, user_key), timeout=20
    )
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not resp.ok:
        message = payload.get("errorMessage") or payload.get("message") or f"HTTP {resp.status_code}"
        raise CopyTradingError(f"eToro live-portfolio lookup failed for {username!r}: {message}")
    return payload.get("positions", [])


def resolve_instruments(instrument_ids: set[int], api_key: str, user_key: str) -> dict[int, tuple[str, str]]:
    """{instrument_id: (symbol, type)} for a batch of instrument ids. Empty
    input returns {} without a network call. Comma-separated ids on a single
    `instrumentsIds` param — confirmed live 2026-09-16; a repeated-param form
    (`instrumentsIds[]`) silently returns the WRONG instruments (ids 1-4 etc.
    regardless of what was asked for) rather than erroring, so this exact
    param shape matters.
    """
    if not instrument_ids:
        return {}
    resp = requests.get(
        INSTRUMENTS_URL,
        headers=_headers(api_key, user_key),
        params={"instrumentsIds": ",".join(str(i) for i in sorted(instrument_ids))},
        timeout=20,
    )
    try:
        payload = resp.json()
    except ValueError:
        payload = {}
    if not resp.ok:
        raise CopyTradingError(f"eToro instrument lookup failed: HTTP {resp.status_code}")
    return {
        row["instrumentId"]: (row.get("symbol"), row.get("type"))
        for row in payload.get("results", [])
    }


def _state_path(username: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
    return COPY_STATE_DIR / f"{safe}.json"


def _load_last_seen(username: str) -> dict[int, dict]:
    path = _state_path(username)
    if not path.exists():
        return {}
    try:
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    except Exception:
        return {}


def _save_last_seen(username: str, positions: dict[int, dict]) -> None:
    COPY_STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(_state_path(username), json.dumps(positions, indent=2))


def poll_investor(username: str, api_key: str, user_key: str) -> list[CopyEvent]:
    """One poll cycle: fetch the investor's current positions, diff against
    what was persisted from the last poll, return the events (if any), and
    persist the new state. First-ever call for a username establishes a
    baseline (every currently-open position is "already known", not
    reported as freshly "opened") — otherwise linking a long-established
    investor would immediately fire a wall of fake open events for
    positions they'd held for months.
    """
    raw_positions = fetch_investor_positions(username, api_key, user_key)
    current_by_id = {int(row["positionId"]): row for row in raw_positions}

    previous_by_id = _load_last_seen(username)
    is_first_poll = not previous_by_id and not _state_path(username).exists()

    instrument_ids = {row["instrumentId"] for row in raw_positions}
    instrument_ids |= {row.get("instrumentId") for row in previous_by_id.values() if row.get("instrumentId")}
    resolved = resolve_instruments(instrument_ids, api_key, user_key)

    def _to_watched(row: dict) -> WatchedPosition:
        symbol, itype = resolved.get(row["instrumentId"], (None, None))
        return WatchedPosition(
            position_id=int(row["positionId"]),
            instrument_id=row["instrumentId"],
            symbol=symbol,
            instrument_type=itype,
            is_buy=bool(row.get("isBuy", True)),
            investment_pct=float(row.get("investmentPct", 0.0)),
            open_timestamp=str(row.get("openTimestamp", "")),
            open_rate=float(row.get("openRate", 0.0)),
        )

    now = datetime.now(timezone.utc).isoformat()
    events: list[CopyEvent] = []

    if not is_first_poll:
        opened_ids = set(current_by_id) - set(previous_by_id)
        closed_ids = set(previous_by_id) - set(current_by_id)
        for pid in opened_ids:
            events.append(CopyEvent("opened", username, now, _to_watched(current_by_id[pid])))
        for pid in closed_ids:
            events.append(CopyEvent("closed", username, now, _to_watched(previous_by_id[pid])))

    _save_last_seen(username, current_by_id)
    return events
