"""Volatility squeeze + breakout: wait for Bollinger Band width to contract
to a recent low (a "squeeze" - the market coiling before a move), then enter
in whichever direction price breaks out, confirmed by MACD histogram
momentum agreeing. Classic "TTM Squeeze"-style combo - genuinely different
from this project's existing BollingerBreakoutStrategy (trades every upper-
band breakout, no volatility-contraction precondition and no momentum
confirmation) and MacdCrossoverStrategy (MACD alone, no volatility context).

Built 2026-09-27 alongside EmaRsiConfirmationStrategy/TripleEmaRibbonStrategy,
same motivation (see EmaRsiConfirmationStrategy's docstring): single-signal
strategies scored a flat 0% real pass rate against the Kraken Funded
challenge sweep - requiring BOTH a genuine volatility contraction AND
momentum confirmation before acting is meant to cut breakout strategies'
usual false-breakout problem.

Band width is measured relative to its own recent range (bottom of its last
`squeeze_lookback` bars, within `squeeze_tolerance`), not a fixed absolute
number - same reasoning DPO Mean-Reversion's own std-dev band already
established: an absolute threshold wouldn't generalize across tickers or
volatility regimes.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class BollingerSqueezeBreakoutStrategy(Strategy):
    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        squeeze_lookback: int = 50,
        squeeze_tolerance: float = 1.10,
        squeeze_recent_bars: int = 5,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
    ):
        self.period = period
        self.num_std = num_std
        self.squeeze_lookback = squeeze_lookback
        self.squeeze_tolerance = squeeze_tolerance
        self.squeeze_recent_bars = squeeze_recent_bars
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal

    def required_lookback(self) -> Lookback:
        return Lookback(bars=max(self.period + self.squeeze_lookback, self.macd_slow + self.macd_signal) + 2)

    def _compute(self, history: pd.DataFrame) -> dict[str, pd.Series]:
        closes = history["close"]
        mid = closes.rolling(self.period).mean()
        std = closes.rolling(self.period).std()
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std
        width = (upper - lower) / mid

        fast_ema = closes.ewm(span=self.macd_fast, adjust=False).mean()
        slow_ema = closes.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        histogram = macd_line - macd_line.ewm(span=self.macd_signal, adjust=False).mean()

        squeeze = width <= width.rolling(self.squeeze_lookback).min() * self.squeeze_tolerance
        return {"closes": closes, "mid": mid, "upper": upper, "lower": lower, "histogram": histogram, "squeeze": squeeze}

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = max(self.period + self.squeeze_lookback, self.macd_slow + self.macd_signal)
        if len(history) < min_len + 1:
            return Signal.HOLD

        c = self._compute(history)
        if pd.isna(c["upper"].iloc[-2]) or pd.isna(c["squeeze"].iloc[-2]) or pd.isna(c["histogram"].iloc[-1]):
            return Signal.HOLD

        prev_close, curr_close = c["closes"].iloc[-2], c["closes"].iloc[-1]
        prev_upper, curr_upper = c["upper"].iloc[-2], c["upper"].iloc[-1]
        curr_mid = c["mid"].iloc[-1]
        curr_hist = c["histogram"].iloc[-1]

        # Squeezed recently, not necessarily on this exact bar - width often
        # starts expanding a few bars before the breakout close actually
        # clears the band, so checking only the current bar would miss most
        # real setups.
        recent_squeeze = bool(c["squeeze"].iloc[-1 - self.squeeze_recent_bars : -1].any())

        breakout_up = prev_close <= prev_upper and curr_close > curr_upper
        reverted_to_mean = curr_close < curr_mid

        if breakout_up and recent_squeeze and curr_hist > 0:
            return Signal.BUY
        if reverted_to_mean:
            return Signal.SELL
        return Signal.HOLD

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        min_len = max(self.period + self.squeeze_lookback, self.macd_slow + self.macd_signal)
        if len(history) < min_len + 1:
            return None
        c = self._compute(history)
        if pd.isna(c["upper"].iloc[-1]):
            return None
        return {
            "upper": float(c["upper"].iloc[-1]),
            "mid": float(c["mid"].iloc[-1]),
            "lower": float(c["lower"].iloc[-1]),
            "macd_histogram": float(c["histogram"].iloc[-1]) if not pd.isna(c["histogram"].iloc[-1]) else 0.0,
        }
