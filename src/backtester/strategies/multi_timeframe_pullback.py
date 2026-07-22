"""Multi-timeframe confirmation trend pullback (EMA-based).

Caveat: this backtester runs one timeframe per scan, so true cross-timeframe
confirmation (e.g. daily trend + 5-minute entry) isn't available here. This
strategy approximates the idea on a single timeframe instead: a slow EMA
stands in for "higher timeframe trend," a fast EMA defines the pullback
entry within that trend. It is not the same as real multi-timeframe
confirmation — flagged here rather than silently treated as equivalent.

Rule: price above the slow EMA = uptrend. Buy when price pulls back down
to (or through) the fast EMA and closes back above it. Sell if price closes
back below the slow EMA (trend invalidated).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import ema, ema_warmup_bars
from backtester.strategy import Bar, Lookback, Signal, Strategy


class MultiTimeframePullbackStrategy(Strategy):
    def __init__(self, trend_span: int = 50, pullback_span: int = 20):
        if pullback_span >= trend_span:
            raise ValueError("pullback_span must be smaller than trend_span")
        self.trend_span = trend_span
        self.pullback_span = pullback_span

    def required_lookback(self) -> Lookback:
        return Lookback(bars=ema_warmup_bars(self.trend_span) + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.trend_span + 2:
            return Signal.HOLD

        closes = history["close"]
        trend_ema = ema(closes, self.trend_span)
        pullback_ema = ema(closes, self.pullback_span)

        in_uptrend = closes.iloc[-1] > trend_ema.iloc[-1]

        prev_close, curr_close = closes.iloc[-2], closes.iloc[-1]
        prev_pullback_ema, curr_pullback_ema = pullback_ema.iloc[-2], pullback_ema.iloc[-1]

        crossed_up_through_pullback_ema = prev_close <= prev_pullback_ema and curr_close > curr_pullback_ema

        if in_uptrend and crossed_up_through_pullback_ema:
            return Signal.BUY

        trend_broken = curr_close < trend_ema.iloc[-1]
        if trend_broken:
            return Signal.SELL

        return Signal.HOLD
