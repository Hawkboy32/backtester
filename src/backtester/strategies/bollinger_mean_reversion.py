"""Bollinger Band mean reversion: the opposite bet from Bollinger Breakout.
Buy when price closes below the lower band (oversold), sell when it reverts
back up to the middle band (the moving average).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class BollingerMeanReversionStrategy(Strategy):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.period + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.period + 1:
            return Signal.HOLD

        closes = history["close"]
        mid = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()
        lower = mid - self.num_std * std

        if pd.isna(lower.iloc[-2]) or pd.isna(mid.iloc[-1]):
            return Signal.HOLD

        prev_close, curr_close = closes.iloc[-2], closes.iloc[-1]
        prev_lower, curr_lower = lower.iloc[-2], lower.iloc[-1]
        curr_mid = mid.iloc[-1]

        broke_down = prev_close >= prev_lower and curr_close < curr_lower
        reverted_to_mean = curr_close > curr_mid

        if broke_down:
            return Signal.BUY
        if reverted_to_mean:
            return Signal.SELL
        return Signal.HOLD
