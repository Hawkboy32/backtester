"""Momentum / rate-of-change strategy: buy when momentum turns positive, sell when it turns negative."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class MomentumRocStrategy(Strategy):
    def __init__(self, period: int = 10, buy_threshold: float = 0.0, sell_threshold: float = 0.0):
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.period + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.period + 2:
            return Signal.HOLD

        closes = history["close"]
        roc = (closes / closes.shift(self.period) - 1) * 100

        if len(roc) < 2 or pd.isna(roc.iloc[-2]) or pd.isna(roc.iloc[-1]):
            return Signal.HOLD

        prev_roc, curr_roc = roc.iloc[-2], roc.iloc[-1]

        crossed_up = prev_roc <= self.buy_threshold and curr_roc > self.buy_threshold
        crossed_down = prev_roc >= self.sell_threshold and curr_roc < self.sell_threshold

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
