"""Fibonacci pullback: within a recent up-swing (N-bar low to N-bar high),
buy when price retraces to a predefined Fibonacci level (default 61.8%) and
bounces — closes back above that level after touching it. Sell if the swing
low is broken instead (the uptrend context is invalidated).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import fibonacci_levels
from backtester.strategy import Bar, Lookback, Signal, Strategy


class FibonacciPullbackStrategy(Strategy):
    def __init__(self, swing_window: int = 30, level: str = "0.618", tolerance_pct: float = 0.3):
        if level not in ("0.236", "0.382", "0.5", "0.618", "0.786"):
            raise ValueError("level must be one of the standard retracement levels")
        self.swing_window = swing_window
        self.level = level
        self.tolerance_pct = tolerance_pct

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.swing_window + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.swing_window + 2:
            return Signal.HOLD

        window = history.iloc[-(self.swing_window + 1) : -1]
        swing_high_val = window["high"].max()
        swing_low_val = window["low"].min()
        if swing_high_val <= swing_low_val:
            return Signal.HOLD

        levels = fibonacci_levels(swing_high_val, swing_low_val)
        target = levels[self.level]
        tolerance = target * (self.tolerance_pct / 100)

        in_uptrend_context = current.close > swing_low_val
        touched_and_bounced = (
            in_uptrend_context and current.low <= target + tolerance and current.close > target
        )
        if touched_and_bounced:
            return Signal.BUY

        broke_support = current.close < swing_low_val
        if broke_support:
            return Signal.SELL

        return Signal.HOLD
