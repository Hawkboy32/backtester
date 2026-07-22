"""VWAP trend following: being above the session VWAP is treated as the
bullish regime. Buy when price crosses up through VWAP, sell when it
crosses back below.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import session_vwap
from backtester.strategy import Bar, Lookback, Signal, Strategy


class VwapTrendStrategy(Strategy):
    def __init__(self, min_bars: int = 5):
        self.min_bars = min_bars

    def required_lookback(self) -> Lookback:
        # 2, not 1: at the first bars of a new session, vwap.iloc[-2] reaches
        # back into the PRIOR session's final VWAP (the series doesn't skip a
        # slot at the day boundary), and the min_bars length gate counts prior-
        # session bars too — sessions=1 would change both behaviors.
        return Lookback(sessions=2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.min_bars + 1:
            return Signal.HOLD

        vwap = session_vwap(history)
        closes = history["close"]

        if pd.isna(vwap.iloc[-2]) or pd.isna(vwap.iloc[-1]):
            return Signal.HOLD

        prev_close, curr_close = closes.iloc[-2], closes.iloc[-1]
        prev_vwap, curr_vwap = vwap.iloc[-2], vwap.iloc[-1]

        crossed_up = prev_close <= prev_vwap and curr_close > curr_vwap
        crossed_down = prev_close >= prev_vwap and curr_close < curr_vwap

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
