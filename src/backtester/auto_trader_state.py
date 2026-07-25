"""Shared state for the auto-trader: a control file the dashboard writes and
the background process reads, and a status file the background process
writes and the dashboard reads. File-based coordination keeps this simple
and avoids IPC/sockets — matches the rest of this project's local-first design.

Fail-closed: if the control file is missing, unreadable, or malformed, the
trader treats itself as disarmed (or worse, killed) rather than guessing a
permissive default. A crash or corrupted file must never result in silent
trading.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "auto_trader_state"
CONTROL_PATH = STATE_DIR / "control.json"
STATUS_PATH = STATE_DIR / "status.json"


def atomic_write_text(path: Path, text: str, replace_attempts: int = 20) -> None:
    """Write a state file so a concurrent reader can never see a half-written
    one. Plain `write_text` truncates first and fills after, leaving a window
    where a reader gets invalid JSON — for control.json that used to be read as
    "corrupt" and therefore killed=True, silently stopping a running bot when
    the dashboard merely saved a config. Write to a temp file in the same
    directory, then os.replace (atomic on Windows and POSIX).

    Windows caveat, found by testing this under a hammering reader: os.replace
    raises PermissionError (WinError 5) if the DESTINATION is open in another
    process at that instant, because Python opens files without
    FILE_SHARE_DELETE. A reader's open() lasts microseconds, so retry briefly
    rather than propagating a spurious failure into the dashboard's save path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(replace_attempts):
            try:
                os.replace(tmp_name, path)
                return
            except PermissionError:
                if attempt == replace_attempts - 1:
                    raise
                time.sleep(0.02)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


@dataclass
class AutoTraderControl:
    enabled: bool = False  # armed to trade this cycle
    killed: bool = False  # emergency stop; always overrides enabled
    allow_live: bool = False  # may target live (non-paper) accounts
    account_ids: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    strategy_name: str = ""
    multiplier: int = 1
    timespan: str = "minute"
    poll_interval_seconds: int = 60
    max_trades_per_day: int = 10
    sizing_mode: str = "fixed_dollars"  # matches backtester.execution.SizingMode values
    sizing_value: float = 500.0
    vol_target_enabled: bool = False  # GARCH volatility filter + sizing (see backtester.volatility)
    vol_target_ann: float = 20.0  # target annualized vol %, only used when vol_target_enabled
    use_roster: bool = False  # adaptive promotion/demotion roster (see backtester.roster) instead
    # of the single strategy_name/tickers pair above — those fields are ignored when this is True
    max_drawdown_enabled: bool = False  # account-level circuit breaker (see backtester.account_risk)
    max_drawdown_pct: float = 10.0  # % below peak equity that hard-blocks new entries for that account
    block_event_days: bool = True  # skip NEW entries on known risk-event days (FOMC — see backtester.events).
    # Defaults ON, unlike the other opt-in filters: it's a pure step-out safety net ("don't sell
    # insurance during a flood warning") and only takes effect on a deliberate (re)start.


@dataclass
class AutoTraderStatus:
    running: bool = False
    pid: int | None = None
    last_heartbeat: str | None = None
    trades_today: int = 0
    trades_date: str | None = None  # YYYY-MM-DD the trades_today counter applies to
    last_signal: str | None = None
    last_error: str | None = None


def _ensure_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def load_control_checked(attempts: int = 3) -> tuple[AutoTraderControl, bool]:
    """(control, readable). `readable` is False only when the file exists but
    could not be parsed after `attempts` tries — callers use it to tell a
    GENUINE kill switch apart from an unreadable file, which must never be
    mistaken for a deliberate stop (see main()'s exit condition).
    """
    _ensure_dir()
    if not CONTROL_PATH.exists():
        return AutoTraderControl(), True
    for attempt in range(attempts):
        try:
            data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
            return AutoTraderControl(**data), True
        except Exception:
            # Writes are atomic now, so this should be unreachable in practice;
            # retry anyway before concluding the file is genuinely corrupt.
            if attempt < attempts - 1:
                time.sleep(0.05)
    # Malformed control file -> fail closed: disarmed AND killed, never "trade anyway".
    return AutoTraderControl(enabled=False, killed=True), False


def load_control() -> AutoTraderControl:
    return load_control_checked()[0]


def save_control(control: AutoTraderControl) -> None:
    atomic_write_text(CONTROL_PATH, json.dumps(asdict(control), indent=2))


def load_status() -> AutoTraderStatus:
    _ensure_dir()
    if not STATUS_PATH.exists():
        return AutoTraderStatus()
    try:
        data = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        return AutoTraderStatus(**data)
    except Exception:
        return AutoTraderStatus()


def save_status(status: AutoTraderStatus) -> None:
    atomic_write_text(STATUS_PATH, json.dumps(asdict(status), indent=2))


def trigger_kill_switch() -> None:
    """Immediately disarm and mark killed. Safe to call even if no trader is running."""
    control = load_control()
    control.killed = True
    control.enabled = False
    save_control(control)
