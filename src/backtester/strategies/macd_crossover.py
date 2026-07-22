"""MACD crossover strategy: buy when MACD line crosses above its signal line."""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import ema_warmup_bars
from backtester.strategy import Bar, Lookback, Signal, Strategy


class MacdCrossoverStrategy(Strategy):
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period

    def required_lookback(self) -> Lookback:
        slowest = max(self.fast_period, self.slow_period, self.signal_period)
        return Lookback(bars=ema_warmup_bars(slowest) + self.slow_period + self.signal_period)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = self.slow_period + self.signal_period
        if len(history) < min_len:
            return Signal.HOLD

        closes = history["close"]
        fast_ema = closes.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = closes.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()

        if len(macd_line) < 2:
            return Signal.HOLD

        prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
        prev_signal, curr_signal = signal_line.iloc[-2], signal_line.iloc[-1]

        crossed_up = prev_macd <= prev_signal and curr_macd > curr_signal
        crossed_down = prev_macd >= prev_signal and curr_macd < curr_signal

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
