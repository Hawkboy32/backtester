"""Conservative/Moderate/Aggressive risk dial - single source of truth for
both the dashboard (app.py) and the mobile backend (Mobile_App/backend/
signal_api.py's /risk-preset), so the two surfaces can't drift apart the way
duplicated config would.

sizing_value ranges updated 2026-08-15 after risk_dial_sizing_sweep.py walk-
forward tested 1%-100% per-trade sizing on the 3 (bucket, strategy) groups
with a real measured edge (see CLAUDE_NOTES.txt, "RISK DIAL, PER-TRADE SIZING
AXIS SWEPT 1%-100%"). Sharpe came back FLAT across the whole range - sizing
doesn't create or destroy edge, it only scales return% and drawdown% together,
linearly. So these three values are a risk-tolerance choice the user made
after seeing that curve, not a backtest-derived optimum: Conservative 5%,
Moderate 15%, Aggressive 50% (previously 0.5% / 1% / 2%). Moderate no longer
matches the pre-dial defaults - applying any of these three now changes real
position sizing on an already-armed account, same as it always could, but the
jump from the old values is much larger than before, so treat re-applying a
preset onto a currently-armed account (especially one with allow_live=True)
with the same care as arming it the first time.
"""

from __future__ import annotations

from backtester.auto_trader_state import AutoTraderControl, load_control, save_control

RISK_PRESETS = {
    "Conservative": {
        "sizing_value": 5.0, "vol_target_ann": 10.0,
        "max_drawdown_pct": 7.0, "giveback_enabled": True, "giveback_pct": 15.0,
    },
    "Moderate": {
        "sizing_value": 15.0, "vol_target_ann": 20.0,
        "max_drawdown_pct": 10.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
    "Aggressive": {
        "sizing_value": 50.0, "vol_target_ann": 30.0,
        "max_drawdown_pct": 15.0, "giveback_enabled": False, "giveback_pct": 25.0,
    },
}


# Per-ticker sizing CEILINGS, layered on top of the preset above rather than
# replacing it (2026-08-16, see CLAUDE_NOTES.txt's liquidity-model entries).
# roster_liquidity_efficient_sizing.py swept all 6 active combos across the
# full 1-100% grid, flat vs a liquidity-aware slippage model, and found the
# roster is NOT uniformly liquid - PSKY's return falls to 4% of its
# flat-slippage value by 100% sizing while ZBRA/BIIB barely register any
# liquidity cost even at 100%. One global sizing_value can't reflect that;
# these three tiers can, at NO interpolation - each value is the largest
# tested grid point (1/2/5/10/15/20/30/50/75/100%) where that ticker's
# liquidity-adjusted return was still at least this fraction of its
# flat-slippage value, same "grid values only" discipline as RISK_PRESETS'
# own vol_target_ann:
#   Conservative -> ratio >= 0.98 (loses at most ~2% to liquidity)
#   Moderate     -> ratio >= 0.90 (loses at most ~10% - same bar the sweep
#                   itself used to flag "liquidity drag exceeds 20%")
#   Aggressive   -> ratio >= 0.75 (loses at most ~25%)
# A ticker with no entry here (anything not in the 6 tested) is uncapped -
# untested, not assumed safe.
TICKER_SIZING_CAPS: dict[str, dict[str, float]] = {
    "PSKY": {"Conservative": 10.0, "Moderate": 20.0, "Aggressive": 30.0},
    "Q": {"Conservative": 30.0, "Moderate": 50.0, "Aggressive": 75.0},
    "ZBRA": {"Conservative": 50.0, "Moderate": 100.0, "Aggressive": 100.0},
    "BEN": {"Conservative": 30.0, "Moderate": 50.0, "Aggressive": 100.0},
    "BIIB": {"Conservative": 75.0, "Moderate": 100.0, "Aggressive": 100.0},
    "MPWR": {"Conservative": 30.0, "Moderate": 75.0, "Aggressive": 100.0},
}


def ticker_sizing_cap(ticker: str, risk_preset: str | None) -> float | None:
    """The liquidity-aware ceiling (% of equity) for this ticker at this risk
    tier, or None if the ticker wasn't part of the liquidity sweep (uncapped
    - untested, not assumed safe, so this never invents a number for a
    ticker it has no evidence about). A preset name outside Conservative/
    Moderate/Aggressive (None, or a custom sizing_value with no preset
    applied) falls back to the Moderate cap - the safety ceiling should
    still apply to a hand-typed sizing_value, and Moderate is the sensible
    middle default, not the most permissive Aggressive one."""
    caps = TICKER_SIZING_CAPS.get(ticker)
    if caps is None:
        return None
    return caps.get(risk_preset, caps["Moderate"])


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
