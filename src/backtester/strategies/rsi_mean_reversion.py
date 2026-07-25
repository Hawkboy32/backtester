"""RSI mean-reversion strategy: buy when oversold, sell when overbought."""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


def _rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


class RsiMeanReversionStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        if not (0 < oversold < overbought < 100):
            raise ValueError("require 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.period + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.period + 1:
            return Signal.HOLD

        rsi = _rsi(history["close"], self.period)
        if len(rsi) < 2:
            return Signal.HOLD

        prev_rsi, curr_rsi = rsi.iloc[-2], rsi.iloc[-1]

        crossed_up_from_oversold = prev_rsi <= self.oversold and curr_rsi > self.oversold
        crossed_down_from_overbought = prev_rsi >= self.overbought and curr_rsi < self.overbought

        if crossed_up_from_oversold:
            return Signal.BUY
        if crossed_down_from_overbought:
            return Signal.SELL
        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """Depth of the oversold dip just before the bounce: how far the prior
        bar's RSI was below the oversold line, as a fraction of oversold. A deeper
        oversold before crossing up = stronger mean-reversion setup. (#25 — logged
        only.)"""
        if len(history) < self.period + 1:
            return None
        rsi = _rsi(history["close"], self.period)
        if len(rsi) < 2:
            return None
        prev_rsi = rsi.iloc[-2]
        if pd.isna(prev_rsi):
            return None
        return (self.oversold - prev_rsi) / self.oversold
