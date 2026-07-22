"""Simple moving-average crossover strategy: buy on golden cross, sell on death cross."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class SmaCrossoverStrategy(Strategy):
    def __init__(self, fast_window: int = 20, slow_window: int = 50):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.slow_window + 1)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.slow_window:
            return Signal.HOLD

        closes = history["close"]
        fast_ma = closes.rolling(self.fast_window).mean()
        slow_ma = closes.rolling(self.slow_window).mean()

        if len(fast_ma) < 2 or pd.isna(fast_ma.iloc[-2]) or pd.isna(slow_ma.iloc[-2]):
            return Signal.HOLD

        prev_fast, prev_slow = fast_ma.iloc[-2], slow_ma.iloc[-2]
        curr_fast, curr_slow = fast_ma.iloc[-1], slow_ma.iloc[-1]

        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
