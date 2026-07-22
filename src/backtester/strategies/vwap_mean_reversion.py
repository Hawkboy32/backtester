"""VWAP mean reversion: the opposite bet from VWAP trend following. Buy when
price has stretched meaningfully below session VWAP (oversold relative to
the day's volume-weighted average), sell when it reverts back up to VWAP.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import session_vwap
from backtester.strategy import Bar, Lookback, Signal, Strategy


class VwapMeanReversionStrategy(Strategy):
    def __init__(self, min_bars: int = 5, entry_deviation_pct: float = 0.5):
        self.min_bars = min_bars
        self.entry_deviation_pct = entry_deviation_pct  # % below VWAP that triggers a buy

    def required_lookback(self) -> Lookback:
        # 2, not 1 — same reasoning as VwapTrendStrategy: iloc[-2] crosses the
        # session boundary at the first bars of each day, and the min_bars
        # gate counts prior-session bars.
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

        prev_deviation_pct = (prev_close - prev_vwap) / prev_vwap * 100
        curr_deviation_pct = (curr_close - curr_vwap) / curr_vwap * 100

        # entering the oversold zone -> buy the dip
        entered_oversold = (
            prev_deviation_pct > -self.entry_deviation_pct
            and curr_deviation_pct <= -self.entry_deviation_pct
        )
        # reverted back up to (or through) VWAP -> take profit
        reverted_to_vwap = prev_close < prev_vwap and curr_close >= curr_vwap

        if entered_oversold:
            return Signal.BUY
        if reverted_to_vwap:
            return Signal.SELL
        return Signal.HOLD
