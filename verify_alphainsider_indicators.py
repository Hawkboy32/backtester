"""Correctness check for the three indicators added 2026-09-06 (DMI/ADX,
Detrended Price Oscillator, Linear Regression Channel) against synthetic
data with a KNOWN right answer - cheaper and more conclusive than only
checking them via a real backtest, where a formula bug and "the strategy
just doesn't find edge" look identical from the outside.

Entirely offline - synthetic OHLC series only, no broker/API calls.
Run: python verify_alphainsider_indicators.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtester.strategies.indicators import (
    detrended_price_oscillator, directional_movement_index, linear_regression_channel,
)

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------
# 1. DMI/ADX: a clean, steady uptrend must show +DI persistently above
#    -DI, and ADX must climb well above Wilder's own 25 "trending" line -
#    the exact condition DmiAdxTrendStrategy requires to enter.
# ---------------------------------------------------------------------
n = 200
up_close = pd.Series(100 + np.arange(n) * 0.5)
up_high = up_close + 0.3
up_low = up_close - 0.3
up_bars = pd.DataFrame({"high": up_high, "low": up_low, "close": up_close})

plus_di, minus_di, adx = directional_movement_index(up_bars, period=14)
tail = slice(100, None)  # well past warmup
check(
    "steady uptrend: +DI stays above -DI once warmed up",
    (plus_di[tail] > minus_di[tail]).all(),
)
check(
    f"steady uptrend: ADX climbs above Wilder's own 25 'trending' line (max={adx[tail].max():.1f})",
    adx[tail].max() > 25.0,
)

# A flat, directionless series must NOT show a confirmed trend - ADX should
# stay low, since there's no real directional move for it to detect at all.
# White noise around a FIXED level (no cumsum - a random WALK still wanders
# with sustained local drift; this needs zero persistent bias by
# construction, which only i.i.d. noise around a constant mean gives).
rng_flat = np.random.default_rng(7)
flat_close = pd.Series(100 + rng_flat.normal(0, 0.5, n))
flat_bars = pd.DataFrame({"high": flat_close + 0.2, "low": flat_close - 0.2, "close": flat_close})
_, _, flat_adx = directional_movement_index(flat_bars, period=14)
check(
    f"flat/choppy series: ADX stays below the trending line (max={flat_adx[tail].max():.1f}) - "
    "this is THE filter DmiAdxTrendStrategy adds over a bare crossover",
    flat_adx[tail].max() < 25.0,
)

# ---------------------------------------------------------------------
# 2. DPO: a pure sine-wave cycle around a price level with NO trend must
#    produce a DPO that oscillates around zero, in phase with the cycle -
#    that's the entire point of a DETRENDED oscillator.
# ---------------------------------------------------------------------
period = 20
cycle_len = 40  # bars per full sine cycle
cyclical = pd.Series(100 + 5 * np.sin(2 * np.pi * np.arange(n) / cycle_len))
dpo = detrended_price_oscillator(cyclical, period)
valid_dpo = dpo.dropna()
check(
    f"pure cycle, no trend: DPO oscillates around zero (mean={valid_dpo.mean():.3f}, "
    f"close to the cycle's own zero-crossing)",
    abs(valid_dpo.mean()) < 1.0,
)
check(
    f"DPO actually swings (std={valid_dpo.std():.2f}) rather than sitting flat at zero",
    valid_dpo.std() > 1.0,
)

# A pure uptrend with NO cycle should detrend down toward zero - DPO's whole
# job is to remove exactly this, not amplify it.
trend_only = pd.Series(100 + np.arange(n) * 0.5)
dpo_trend = detrended_price_oscillator(trend_only, period).dropna()
check(
    f"pure trend, no cycle: DPO does NOT grow with the trend (last value={dpo_trend.iloc[-1]:.2f}, "
    f"vs. raw price move of {trend_only.iloc[-1] - trend_only.iloc[0]:.1f} over the same span)",
    abs(dpo_trend.iloc[-1]) < 10.0,
)

# ---------------------------------------------------------------------
# 3. Linear Regression Channel: a PERFECTLY straight line has zero
#    residual (the regression fits it exactly), and the fitted line value
#    must equal the actual price. Adding known noise must make
#    residual_std track that noise's own real magnitude.
# ---------------------------------------------------------------------
lrc_period = 20
straight = pd.Series(100 + np.arange(n) * 0.5)
line, resid_std = linear_regression_channel(straight, lrc_period)
check(
    f"perfectly straight line: regression fits it exactly (max resid_std={resid_std.dropna().max():.6f})",
    resid_std.dropna().max() < 1e-6,
)
check(
    f"perfectly straight line: fitted line value equals actual price (max error="
    f"{(line.dropna() - straight[line.notna()]).abs().max():.6f})",
    (line.dropna() - straight[line.notna()]).abs().max() < 1e-6,
)

rng = np.random.default_rng(42)
noisy = pd.Series(100 + np.arange(n) * 0.5 + rng.normal(0, 2.0, n))  # true noise std = 2.0
_, noisy_resid_std = linear_regression_channel(noisy, lrc_period)
measured = noisy_resid_std.dropna().mean()
check(
    f"known noise (true std=2.0): measured residual_std ({measured:.2f}) is in a sane "
    "range for a 20-bar window (not exact - small-sample regression bias is expected)",
    1.0 < measured < 3.5,
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All checks passed.")
