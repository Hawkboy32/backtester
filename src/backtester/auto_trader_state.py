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
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATE_DIR = Path(__file__).resolve().parent.parent.parent / "auto_trader_state"
CONTROL_PATH = STATE_DIR / "control.json"
STATUS_PATH = STATE_DIR / "status.json"


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


def load_control() -> AutoTraderControl:
    _ensure_dir()
    if not CONTROL_PATH.exists():
        return AutoTraderControl()
    try:
        data = json.loads(CONTROL_PATH.read_text(encoding="utf-8"))
        return AutoTraderControl(**data)
    except Exception:
        # Malformed control file -> fail closed: disarmed AND killed, never "trade anyway".
        return AutoTraderControl(enabled=False, killed=True)


def save_control(control: AutoTraderControl) -> None:
    _ensure_dir()
    CONTROL_PATH.write_text(json.dumps(asdict(control), indent=2), encoding="utf-8")


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
    _ensure_dir()
    STATUS_PATH.write_text(json.dumps(asdict(status), indent=2), encoding="utf-8")


def trigger_kill_switch() -> None:
    """Immediately disarm and mark killed. Safe to call even if no trader is running."""
    control = load_control()
    control.killed = True
    control.enabled = False
    save_control(control)
