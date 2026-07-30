"""Local store for saved Backtest/Scanner form configurations, so a setup
can be re-run later without rebuilding it from scratch. No secrets live
here — just tickers, dates, strategy choices, and similar form values.
"""

from __future__ import annotations

import json
from pathlib import Path

from backtester.auto_trader_state import atomic_write_text

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "saved_configs.json"


def _load_all() -> dict:
    if not CONFIGS_PATH.exists():
        return {"backtest": {}, "scanner": {}}
    try:
        data = json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        # Fail closed like every other local state store in this project
        # (app_settings/roster/heartbeat/account_risk/two_factor) — a corrupt
        # file degrades to "no saved configs" instead of crashing the
        # Backtest/Scanner tab. Found 2026-07-26 during a full-app QA pass:
        # this was the one store that didn't follow that pattern.
        return {"backtest": {}, "scanner": {}}
    data.setdefault("backtest", {})
    data.setdefault("scanner", {})
    return data


def _save_all(data: dict) -> None:
    atomic_write_text(CONFIGS_PATH, json.dumps(data, indent=2))


def list_configs(kind: str) -> list[str]:
    return sorted(_load_all().get(kind, {}).keys())


def save_config(kind: str, name: str, params: dict) -> None:
    data = _load_all()
    data.setdefault(kind, {})[name] = params
    _save_all(data)


def load_config(kind: str, name: str) -> dict | None:
    return _load_all().get(kind, {}).get(name)


def delete_config(kind: str, name: str) -> None:
    data = _load_all()
    data.get(kind, {}).pop(name, None)
    _save_all(data)
