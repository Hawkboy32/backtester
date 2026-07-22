"""Break-and-retest: price breaks above a recent swing-high resistance
level, then pulls back to retest that same level (now acting as support)
and closes back above it — confirming the retest held. Buy on a confirmed
retest; sell if price instead breaks back below the old resistance.

Well-defined ruleset (there's no single universal definition of this
pattern, so this is the specific version implemented here):
1. Resistance = rolling N-bar high, evaluated one bar at a time.
2. A "breakout" is a bar whose close crosses above that resistance level.
3. Within the following `retest_lookback` bars, if price dips back to
   within `retest_tolerance_pct` of the breakout level and closes above
   it, that's a confirmed retest -> buy.
4. If instead price closes back below the breakout level (beyond
   tolerance), the retest failed -> sell.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import swing_high
from backtester.strategy import Bar, Lookback, Signal, Strategy


class BreakAndRetestStrategy(Strategy):
    def __init__(self, swing_window: int = 20, retest_lookback: int = 15, retest_tolerance_pct: float = 0.3):
        self.swing_window = swing_window
        self.retest_lookback = retest_lookback
        self.retest_tolerance_pct = retest_tolerance_pct

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.swing_window + self.retest_lookback + 3)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = self.swing_window + self.retest_lookback + 2
        if len(history) < min_len:
            return Signal.HOLD

        highs = history["high"]
        closes = history["close"]
        resistance = swing_high(highs, self.swing_window).shift(1)

        recent = history.iloc[-(self.retest_lookback + 1) : -1]
        recent_resistance = resistance.loc[recent.index]
        recent_closes = closes.loc[recent.index]
        prev_closes = recent_closes.shift(1)
        prev_resistance = recent_resistance.shift(1)
        breakout_mask = (prev_closes <= prev_resistance) & (recent_closes > recent_resistance)

        if not breakout_mask.any():
            return Signal.HOLD

        breakout_level = recent_resistance[breakout_mask].iloc[-1]
        if pd.isna(breakout_level):
            return Signal.HOLD

        tolerance = breakout_level * (self.retest_tolerance_pct / 100)

        retested = current.low <= breakout_level + tolerance and current.close > breakout_level
        if retested:
            return Signal.BUY

        broke_back_below = current.close < breakout_level - tolerance
        if broke_back_below:
            return Signal.SELL

        return Signal.HOLD
