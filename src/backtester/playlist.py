"""Queue of scan configurations to run unattended — the "learning dataset"
firehose.

Why this exists: a single Scanner run is synchronous in the browser tab and a
50-ticker x 18-strategy scan took ~4.5 hours, so accumulating a broad, diverse
backtest dataset meant babysitting the tab one scan at a time. A playlist lets
several scan configs be queued and worked through by a standalone process, the
same architecture as auto_trader.py (and for the same reason — Streamlit's
rerun-per-interaction model is not suited to hours-long work that must survive
the tab closing).

State lives in two gitignored files next to the other local state:
  playlist.json  — the queue itself (written by the dashboard AND the runner)
  playlist_status.json — the runner's heartbeat/progress (written by the runner)

Every item's results land in scan_history.db exactly like a manual scan, so
Scan History stays the single place results accumulate.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text

PLAYLIST_PATH = STATE_DIR / "playlist.json"
PLAYLIST_STATUS_PATH = STATE_DIR / "playlist_status.json"

# An item is queued -> running -> done | failed | cancelled.
PENDING, RUNNING, DONE, FAILED, CANCELLED = "pending", "running", "done", "failed", "cancelled"
TERMINAL = {DONE, FAILED, CANCELLED}


@dataclass
class PlaylistItem:
    """One queued scan. Mirrors run_scan's parameters so the runner can hand
    them straight over without a translation layer drifting out of sync."""

    label: str = ""
    universe: str = "S&P 500"
    max_tickers: int = 25
    strategy_names: list[str] = field(default_factory=list)
    from_date: str = ""
    to_date: str = ""
    multiplier: int = 1
    timespan: str = "minute"
    starting_cash: float = 100_000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 2.0
    requests_per_minute: int = 5
    max_workers: int = 4
    vol_target_enabled: bool = False
    target_vol_ann: float = 20.0
    event_filter_enabled: bool = False

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: str = PENDING
    queued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    run_id: int | None = None        # scan_history.db run this produced
    num_results: int | None = None
    error: str | None = None

    def describe(self) -> str:
        return self.label or (
            f"{self.universe} x{self.max_tickers} · {len(self.strategy_names)} strategies · "
            f"{self.from_date}→{self.to_date} · {self.multiplier}{self.timespan[:1]}"
        )


@dataclass
class PlaylistStatus:
    running: bool = False
    pid: int | None = None
    last_heartbeat: str | None = None
    current_item_id: str | None = None
    current_label: str | None = None
    current_ticker: str | None = None
    current_progress: float = 0.0     # 0..1 within the current item
    items_done: int = 0
    items_total: int = 0
    last_error: str | None = None
    stop_requested: bool = False      # dashboard asks the runner to stop after the current ticker


def _ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_playlist() -> list[PlaylistItem]:
    _ensure_dir()
    if not PLAYLIST_PATH.exists():
        return []
    try:
        data = json.loads(PLAYLIST_PATH.read_text(encoding="utf-8"))
    except Exception:
        # A queue we can't read must not silently look empty to the dashboard;
        # but it also isn't safety-critical (nothing trades), so return empty
        # and let the caller surface it. The runner logs it via last_error.
        return []
    items: list[PlaylistItem] = []
    for raw in data.get("items", []):
        # Tolerate items written by an older/newer version rather than losing
        # the whole queue over one unknown field.
        known = {k: v for k, v in raw.items() if k in PlaylistItem.__dataclass_fields__}
        items.append(PlaylistItem(**known))
    return items


def save_playlist(items: list[PlaylistItem]) -> None:
    atomic_write_text(PLAYLIST_PATH, json.dumps({"items": [asdict(i) for i in items]}, indent=2))


def load_status() -> PlaylistStatus:
    _ensure_dir()
    if not PLAYLIST_STATUS_PATH.exists():
        return PlaylistStatus()
    try:
        data = json.loads(PLAYLIST_STATUS_PATH.read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in PlaylistStatus.__dataclass_fields__}
        return PlaylistStatus(**known)
    except Exception:
        return PlaylistStatus()


def save_status(status: PlaylistStatus) -> None:
    atomic_write_text(PLAYLIST_STATUS_PATH, json.dumps(asdict(status), indent=2))


def add_item(item: PlaylistItem) -> None:
    items = load_playlist()
    items.append(item)
    save_playlist(items)


def remove_item(item_id: str) -> None:
    save_playlist([i for i in load_playlist() if i.id != item_id])


def move_item(item_id: str, delta: int) -> None:
    """Reorder within the queue. Only meaningful for pending items."""
    items = load_playlist()
    idx = next((n for n, i in enumerate(items) if i.id == item_id), None)
    if idx is None:
        return
    new_idx = max(0, min(len(items) - 1, idx + delta))
    if new_idx == idx:
        return
    items.insert(new_idx, items.pop(idx))
    save_playlist(items)


def update_item(item_id: str, **changes) -> None:
    """Re-read, mutate, write. The runner and dashboard both write this file,
    so always work from the file's current contents rather than a stale copy —
    otherwise a dashboard edit could clobber the runner's progress or vice versa."""
    items = load_playlist()
    for item in items:
        if item.id == item_id:
            for k, v in changes.items():
                setattr(item, k, v)
    save_playlist(items)


def next_pending(items: list[PlaylistItem] | None = None) -> PlaylistItem | None:
    for item in items if items is not None else load_playlist():
        if item.status == PENDING:
            return item
    return None


def reset_item(item_id: str) -> None:
    """Put a finished/failed item back in the queue to run again."""
    update_item(item_id, status=PENDING, started_at=None, finished_at=None,
                run_id=None, num_results=None, error=None)


def request_stop(stop: bool = True) -> None:
    status = load_status()
    status.stop_requested = stop
    save_status(status)


def clear_finished() -> None:
    save_playlist([i for i in load_playlist() if i.status not in TERMINAL])
