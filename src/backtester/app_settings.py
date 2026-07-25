"""Local store for dashboard-wide default settings that used to live scattered
across individual tabs. Currently holds the execution-cost defaults (commission
per trade + slippage in bps) that new Backtest/Scanner runs start from, set in
one place on the Settings page instead of being re-entered per tab.

No secrets here — just default form values. File lives at the project root
alongside saved_configs.json, and is gitignored (a local preference, not code).
"""

from __future__ import annotations

import json
from pathlib import Path

SETTINGS_PATH = Path(__file__).resolve().parent.parent.parent / "app_settings.json"

# Defaults model Alpaca: $0 commission on US stocks/ETFs, ~2 bps slippage/side
# (spread-crossing + sub-bp sell-side regulatory fees). Same values the Backtest
# and Scanner tabs hardcoded before these moved to the Settings page.
DEFAULT_COMMISSION = 0.0
DEFAULT_SLIPPAGE_BPS = 2.0


def load_settings() -> dict:
    """Return {commission, slippage_bps}, falling back to the Alpaca-modelled
    defaults for a missing/corrupt file (never raises)."""
    defaults = {"commission": DEFAULT_COMMISSION, "slippage_bps": DEFAULT_SLIPPAGE_BPS}
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return {
            "commission": float(data.get("commission", DEFAULT_COMMISSION)),
            "slippage_bps": float(data.get("slippage_bps", DEFAULT_SLIPPAGE_BPS)),
        }
    except Exception:
        return defaults


def save_settings(cfg: dict) -> None:
    payload = {
        "commission": float(cfg.get("commission", DEFAULT_COMMISSION)),
        "slippage_bps": float(cfg.get("slippage_bps", DEFAULT_SLIPPAGE_BPS)),
    }
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
