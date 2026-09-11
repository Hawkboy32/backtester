"""Linear Regression Channel mean reversion: same shape as Bollinger Mean
Reversion (buy an oversold break, sell on reversion to the middle), but the
middle line is a rolling OLS regression through recent price rather than a
flat SMA - it tilts with the trend instead of assuming price reverts to a
flat average, which matters most on a trending ticker where a flat SMA
middle band drifts persistently behind price.

Sourced from AlphaInsider strategy-browsing (2026-09-06) as a candidate worth
testing - "Linear Regression Channel" is a standard charting construct
(rolling OLS line ± a multiple of residual std-dev), not any one script
author's proprietary method.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import linear_regression_channel
from backtester.strategy import Bar, Lookback, Signal, Strategy


class LinearRegressionChannelStrategy(Strategy):
    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.period + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.period + 1:
            return Signal.HOLD

        closes = history["close"]
        line, resid_std = linear_regression_channel(closes, self.period)
        lower = line - self.num_std * resid_std

        if pd.isna(lower.iloc[-2]) or pd.isna(line.iloc[-1]):
            return Signal.HOLD

        prev_close, curr_close = closes.iloc[-2], closes.iloc[-1]
        prev_lower, curr_lower = lower.iloc[-2], lower.iloc[-1]
        curr_line = line.iloc[-1]

        broke_down = prev_close >= prev_lower and curr_close < curr_lower
        reverted_to_line = curr_close > curr_line

        if broke_down:
            return Signal.BUY
        if reverted_to_line:
            return Signal.SELL
        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far below the lower channel the close is, as a fraction of the
        channel half-width - same shape as BollingerMeanReversionStrategy's
        own conviction (#25 - logged only, does not affect sizing)."""
        if len(history) < self.period + 1:
            return None
        closes = history["close"]
        line, resid_std = linear_regression_channel(closes, self.period)
        curr_line, curr_resid_std, curr_close = line.iloc[-1], resid_std.iloc[-1], closes.iloc[-1]
        if pd.isna(curr_line) or pd.isna(curr_resid_std):
            return None
        half_width = self.num_std * curr_resid_std
        if half_width <= 0:
            return None
        return (curr_line - half_width - curr_close) / half_width

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        if len(history) < self.period + 1:
            return None
        closes = history["close"]
        line, resid_std = linear_regression_channel(closes, self.period)
        curr_line, curr_resid_std = line.iloc[-1], resid_std.iloc[-1]
        if pd.isna(curr_line) or pd.isna(curr_resid_std):
            return None
        return {
            "line": float(curr_line),
            "lower": float(curr_line - self.num_std * curr_resid_std),
            "upper": float(curr_line + self.num_std * curr_resid_std),
            "num_std": self.num_std,
        }
