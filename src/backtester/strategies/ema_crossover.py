"""EMA crossover momentum: buy when the fast EMA crosses above the slow EMA,
sell when it crosses back below. Same idea as SMA Crossover, but EMA weights
recent bars more heavily, so it reacts faster to new momentum.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import ema, ema_warmup_bars
from backtester.strategy import Bar, Lookback, Signal, Strategy


class EmaCrossoverStrategy(Strategy):
    def __init__(self, fast_span: int = 12, slow_span: int = 26):
        if fast_span >= slow_span:
            raise ValueError("fast_span must be smaller than slow_span")
        self.fast_span = fast_span
        self.slow_span = slow_span

    def required_lookback(self) -> Lookback:
        return Lookback(bars=ema_warmup_bars(self.slow_span) + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.slow_span + 1:
            return Signal.HOLD

        closes = history["close"]
        fast = ema(closes, self.fast_span)
        slow = ema(closes, self.slow_span)

        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]

        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
