"""EMA crossover gated by RSI directional confirmation: only take a fast/slow
EMA crossover when RSI agrees with the same direction (>50 for a buy, <50
for a sell) - a standard trend-following technique for cutting the false
crossover signals a bare moving-average cross is known to throw in chop.

Built 2026-09-27 after EMA-adjacent strategies (MACD Crossover) and other
single-signal strategies scored a flat 0% real pass rate against the Kraken
Funded challenge sizing sweep - motivated by the theory that the failure
mode is false/whipsaw crossover signals blowing through the challenge's -3%
floor, which a second, independent confirming signal should reduce. Distinct
from this project's existing EmaCrossoverStrategy (no filter at all) and
RsiMeanReversionStrategy (RSI drives entries directly, not just a filter).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import ema, ema_warmup_bars, rsi
from backtester.strategy import Bar, Lookback, Signal, Strategy


class EmaRsiConfirmationStrategy(Strategy):
    def __init__(self, fast_span: int = 12, slow_span: int = 26, rsi_period: int = 14, rsi_midline: float = 50.0):
        if fast_span >= slow_span:
            raise ValueError("fast_span must be smaller than slow_span")
        self.fast_span = fast_span
        self.slow_span = slow_span
        self.rsi_period = rsi_period
        self.rsi_midline = rsi_midline

    def required_lookback(self) -> Lookback:
        return Lookback(bars=max(ema_warmup_bars(self.slow_span), self.rsi_period) + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = max(self.slow_span, self.rsi_period) + 1
        if len(history) < min_len:
            return Signal.HOLD

        closes = history["close"]
        fast = ema(closes, self.fast_span)
        slow = ema(closes, self.slow_span)
        rsi_series = rsi(closes, self.rsi_period)

        if len(fast) < 2 or pd.isna(fast.iloc[-2]) or pd.isna(rsi_series.iloc[-1]):
            return Signal.HOLD

        prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
        curr_fast, curr_slow = fast.iloc[-1], slow.iloc[-1]
        curr_rsi = rsi_series.iloc[-1]

        crossed_up = prev_fast <= prev_slow and curr_fast > curr_slow
        crossed_down = prev_fast >= prev_slow and curr_fast < curr_slow

        # RSI confirmation only gates the ENTRY, not the exit - same
        # "a close always falls through untouched" principle DmiAdxTrendStrategy
        # already uses for its own filter, so a confirmed position isn't
        # trapped open if RSI has since drifted back across the midline.
        if crossed_up and curr_rsi > self.rsi_midline:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        min_len = max(self.slow_span, self.rsi_period) + 1
        if len(history) < min_len:
            return None
        closes = history["close"]
        fast, slow = ema(closes, self.fast_span), ema(closes, self.slow_span)
        curr_rsi = rsi(closes, self.rsi_period).iloc[-1]
        if pd.isna(fast.iloc[-1]) or pd.isna(curr_rsi):
            return None
        return {"fast_ema": float(fast.iloc[-1]), "slow_ema": float(slow.iloc[-1]), "rsi": float(curr_rsi)}
