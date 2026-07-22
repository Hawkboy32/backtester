"""Local store for saved Backtest/Scanner form configurations, so a setup
can be re-run later without rebuilding it from scratch. No secrets live
here — just tickers, dates, strategy choices, and similar form values.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIGS_PATH = Path(__file__).resolve().parent.parent.parent / "saved_configs.json"


def _load_all() -> dict:
    if not CONFIGS_PATH.exists():
        return {"backtest": {}, "scanner": {}}
    data = json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
    data.setdefault("backtest", {})
    data.setdefault("scanner", {})
    return data


def _save_all(data: dict) -> None:
    CONFIGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


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
