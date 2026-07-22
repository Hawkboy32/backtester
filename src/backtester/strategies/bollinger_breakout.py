"""Bollinger Band breakout strategy: buy on upper-band breakout, exit at the mean."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class BollingerBreakoutStrategy(Strategy):
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
        upper = mid + self.num_std * std

        if pd.isna(upper.iloc[-2]) or pd.isna(mid.iloc[-1]):
            return Signal.HOLD

        prev_close, curr_close = closes.iloc[-2], closes.iloc[-1]
        prev_upper, curr_upper = upper.iloc[-2], upper.iloc[-1]
        curr_mid = mid.iloc[-1]

        breakout_up = prev_close <= prev_upper and curr_close > curr_upper
        reverted_to_mean = curr_close < curr_mid

        if breakout_up:
            return Signal.BUY
        if reverted_to_mean:
            return Signal.SELL
        return Signal.HOLD
