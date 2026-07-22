"""RSI divergence: bullish divergence is when price makes a lower low but
RSI makes a higher low at that same point — momentum is fading even as
price keeps falling. Buy once price confirms by closing back above the
more recent swing low. Exit when RSI reaches overbought.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import rsi
from backtester.strategy import Bar, Lookback, Signal, Strategy


class RsiDivergenceStrategy(Strategy):
    def __init__(self, rsi_period: int = 14, lookback: int = 10, overbought: float = 70.0):
        self.rsi_period = rsi_period
        self.lookback = lookback
        self.overbought = overbought

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.rsi_period + self.lookback * 2 + 3)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = self.rsi_period + self.lookback * 2 + 2
        if len(history) < min_len:
            return Signal.HOLD

        closes = history["close"]
        rsi_series = rsi(closes, self.rsi_period)

        recent_window = history.iloc[-self.lookback :]
        prior_window = history.iloc[-(self.lookback * 2) : -self.lookback]

        recent_closes = closes.loc[recent_window.index]
        prior_closes = closes.loc[prior_window.index]
        recent_rsi = rsi_series.loc[recent_window.index]
        prior_rsi = rsi_series.loc[prior_window.index]

        recent_low_idx = recent_closes.idxmin()
        prior_low_idx = prior_closes.idxmin()

        recent_low_price = recent_closes.loc[recent_low_idx]
        prior_low_price = prior_closes.loc[prior_low_idx]
        recent_low_rsi = recent_rsi.loc[recent_low_idx]
        prior_low_rsi = prior_rsi.loc[prior_low_idx]

        if pd.isna(recent_low_rsi) or pd.isna(prior_low_rsi):
            return Signal.HOLD

        bullish_divergence = recent_low_price < prior_low_price and recent_low_rsi > prior_low_rsi
        confirmed = bullish_divergence and current.close > recent_low_price

        if confirmed:
            return Signal.BUY
        if rsi_series.iloc[-1] > self.overbought:
            return Signal.SELL
        return Signal.HOLD
