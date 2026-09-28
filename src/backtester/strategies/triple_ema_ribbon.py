"""Triple EMA ribbon: buy when fast > mid > slow EMA first all align in
bullish order (not already aligned the bar before), sell when that order
breaks. A well-known trend-confirmation technique - three independent
moving averages agreeing on direction is a stronger trend signal than any
one crossover alone.

Built 2026-09-27 alongside EmaRsiConfirmationStrategy, same motivation (see
that file's docstring): single-signal crossover strategies scored a flat 0%
real pass rate against the Kraken Funded challenge sweep. Genuinely
different from this project's existing EmaCrossoverStrategy - that's a
single 2-EMA cross; this requires three EMAs to agree, which a single
crossover can satisfy while the broader trend is still ambiguous.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import ema, ema_warmup_bars
from backtester.strategy import Bar, Lookback, Signal, Strategy


class TripleEmaRibbonStrategy(Strategy):
    def __init__(self, fast_span: int = 9, mid_span: int = 21, slow_span: int = 55):
        if not (fast_span < mid_span < slow_span):
            raise ValueError("require fast_span < mid_span < slow_span")
        self.fast_span = fast_span
        self.mid_span = mid_span
        self.slow_span = slow_span

    def required_lookback(self) -> Lookback:
        return Lookback(bars=ema_warmup_bars(self.slow_span) + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.slow_span + 1:
            return Signal.HOLD

        closes = history["close"]
        fast = ema(closes, self.fast_span)
        mid = ema(closes, self.mid_span)
        slow = ema(closes, self.slow_span)

        if len(fast) < 2 or pd.isna(fast.iloc[-2]) or pd.isna(slow.iloc[-1]):
            return Signal.HOLD

        prev_aligned_up = fast.iloc[-2] > mid.iloc[-2] > slow.iloc[-2]
        curr_aligned_up = fast.iloc[-1] > mid.iloc[-1] > slow.iloc[-1]

        # Fire once, on the bar the ribbon FIRST aligns - not on every bar it
        # stays aligned, same "fire once" reasoning VwapMeanReversionStrategy
        # already uses for its own acceptance-run logic.
        entered_alignment = curr_aligned_up and not prev_aligned_up
        broke_alignment = prev_aligned_up and not curr_aligned_up

        if entered_alignment:
            return Signal.BUY
        if broke_alignment:
            return Signal.SELL
        return Signal.HOLD

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        if len(history) < self.slow_span + 1:
            return None
        closes = history["close"]
        fast, mid, slow = ema(closes, self.fast_span), ema(closes, self.mid_span), ema(closes, self.slow_span)
        if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]):
            return None
        return {"fast_ema": float(fast.iloc[-1]), "mid_ema": float(mid.iloc[-1]), "slow_ema": float(slow.iloc[-1])}
