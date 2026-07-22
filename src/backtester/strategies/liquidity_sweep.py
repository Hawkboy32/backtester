"""Liquidity sweep / stop hunt (bullish variant, long-only): price spikes
below a recent swing low — sweeping stop-loss orders resting there — then
closes back above it within the same bar, suggesting the breakdown was a
fakeout rather than a real move. Buy on the reclaim; sell if price later
breaks back below that same level for real.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import swing_low
from backtester.strategy import Bar, Lookback, Signal, Strategy


class LiquiditySweepStrategy(Strategy):
    def __init__(self, swing_window: int = 20):
        self.swing_window = swing_window

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.swing_window + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.swing_window + 2:
            return Signal.HOLD

        prior_low = swing_low(history["low"].iloc[:-1], self.swing_window).iloc[-1]
        if pd.isna(prior_low):
            return Signal.HOLD

        swept_and_reclaimed = current.low < prior_low and current.close > prior_low
        broke_down_again = current.close < prior_low

        if swept_and_reclaimed:
            return Signal.BUY
        if broke_down_again:
            return Signal.SELL
        return Signal.HOLD
