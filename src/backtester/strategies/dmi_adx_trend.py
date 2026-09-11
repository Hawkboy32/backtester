"""DMI/ADX trend strategy: buy when +DI crosses above -DI while ADX confirms
a real trend is underway (not just noise); sell on the opposite crossover.

Sourced from AlphaInsider strategy-browsing (2026-09-06) as a candidate worth
testing - Wilder's DMI/ADX (New Concepts in Technical Trading Systems, 1978)
is the standard, textbook formula, not any one script author's version. The
genuinely new piece versus this project's existing trend strategies (SMA/EMA/
MACD crossover) is the ADX filter itself: none of them distinguish a real
trend from a directionless chop before acting on a crossover.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import directional_movement_index
from backtester.strategy import Bar, Lookback, Signal, Strategy


class DmiAdxTrendStrategy(Strategy):
    def __init__(self, period: int = 14, adx_threshold: float = 25.0):
        # 25 is Wilder's own published line for "trending" vs. "flat" market -
        # not tuned here, the same reference value the indicator was
        # introduced with.
        self.period = period
        self.adx_threshold = adx_threshold

    def required_lookback(self) -> Lookback:
        # Wilder's smoothing is an EWM approximation (see indicators.py), so
        # like MACD's ema_warmup_bars there's no exact finite window where
        # it's "converged" - a generous multiple of the period is the same
        # practical tradeoff this project already makes for EWM-based
        # indicators elsewhere.
        return Lookback(bars=self.period * 6)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.period * 3:
            return Signal.HOLD

        plus_di, minus_di, adx = directional_movement_index(history, self.period)
        if len(plus_di) < 2 or pd.isna(plus_di.iloc[-2]) or pd.isna(adx.iloc[-1]):
            return Signal.HOLD

        prev_plus, curr_plus = plus_di.iloc[-2], plus_di.iloc[-1]
        prev_minus, curr_minus = minus_di.iloc[-2], minus_di.iloc[-1]
        curr_adx = adx.iloc[-1]

        crossed_up = prev_plus <= prev_minus and curr_plus > curr_minus
        crossed_down = prev_plus >= prev_minus and curr_plus < curr_minus

        # The ADX filter only gates NEW entries - same "a close always falls
        # through untouched" principle already used for event-day/storm
        # blocks in auto_trader.py, so a real trend that's still confirmed
        # can still be exited on the opposite crossover even if ADX has since
        # dipped, rather than trapping the position open.
        if crossed_up and curr_adx >= self.adx_threshold:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far ADX sits above the trend threshold, scaled against a
        generous 60-point ceiling (ADX rarely sustains above the 40s-50s in
        practice) - stronger trend confirmation = higher conviction."""
        if len(history) < self.period * 3:
            return None
        _, _, adx = directional_movement_index(history, self.period)
        curr_adx = adx.iloc[-1]
        if pd.isna(curr_adx):
            return None
        span = 60.0 - self.adx_threshold
        if span <= 0:
            return None
        return max(0.0, min(1.0, (curr_adx - self.adx_threshold) / span))

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        if len(history) < self.period * 3:
            return None
        plus_di, minus_di, adx = directional_movement_index(history, self.period)
        curr_plus, curr_minus, curr_adx = plus_di.iloc[-1], minus_di.iloc[-1], adx.iloc[-1]
        if pd.isna(curr_plus) or pd.isna(curr_minus) or pd.isna(curr_adx):
            return None
        return {
            "plus_di": float(curr_plus),
            "minus_di": float(curr_minus),
            "adx": float(curr_adx),
            "adx_threshold": self.adx_threshold,
        }
