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

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far below the lower band the close is, as a fraction of the band
        half-width (num_std*std). Deeper break = more oversold = higher conviction.
        (#25 — logged only, does not affect sizing.)"""
        if len(history) < self.period + 1:
            return None
        closes = history["close"]
        mid = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()
        lower = mid - self.num_std * std
        curr_mid, curr_lower, curr_close = mid.iloc[-1], lower.iloc[-1], closes.iloc[-1]
        if pd.isna(curr_lower) or pd.isna(curr_mid):
            return None
        half_width = curr_mid - curr_lower  # = num_std * std
        if half_width <= 0:
            return None
        return (curr_lower - curr_close) / half_width

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        """Mid/upper/lower band values this bar - what this strategy is actually
        watching, for display (#detail request)."""
        if len(history) < self.period + 1:
            return None
        closes = history["close"]
        mid = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()
        curr_mid, curr_std = mid.iloc[-1], std.iloc[-1]
        if pd.isna(curr_mid) or pd.isna(curr_std):
            return None
        return {
            "mid": float(curr_mid),
            "lower": float(curr_mid - self.num_std * curr_std),
            "upper": float(curr_mid + self.num_std * curr_std),
            "num_std": self.num_std,
        }
