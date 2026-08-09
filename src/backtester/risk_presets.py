"""Conservative/Moderate/Aggressive risk dial - single source of truth for
both the dashboard (app.py) and the mobile backend (Mobile_App/backend/
signal_api.py's /risk-preset), so the two surfaces can't drift apart the way
duplicated config would. Moderate matches the pre-dial defaults exactly, so
applying it is a no-op for an account already running the defaults.
"""

from __future__ import annotations

from backtester.auto_trader_state import AutoTraderControl, load_control, save_control

RISK_PRESETS = {
    "Conservative": {
        "sizing_value": 0.5, "vol_target_ann": 10.0,
        "max_drawdown_pct": 7.0, "giveback_enabled": True, "giveback_pct": 15.0,
    },
    "Moderate": {
        "sizing_value": 1.0, "vol_target_ann": 20.0,
        "max_drawdown_pct": 10.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
    "Aggressive": {
        "sizing_value": 2.0, "vol_target_ann": 30.0,
        "max_drawdown_pct": 15.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
}


def apply_risk_preset(name: str) -> AutoTraderControl:
    """Load-mutate-save control.json with one preset's values. Pure control-
    state change only - callers that also need to update UI state (the
    dashboard's session_state-backed widgets) do that separately on top of
    this, since that's presentation, not the actual setting."""
    preset = RISK_PRESETS[name]
    control = load_control()
    control.sizing_mode = "pct_equity"
    control.sizing_value = preset["sizing_value"]
    control.vol_target_enabled = True
    control.vol_target_ann = preset["vol_target_ann"]
    control.max_drawdown_enabled = True
    control.max_drawdown_pct = preset["max_drawdown_pct"]
    control.giveback_enabled = preset["giveback_enabled"]
    control.giveback_pct = preset["giveback_pct"]
    control.risk_preset = name
    save_control(control)
    return control
