"""VWAP mean reversion: the opposite bet from VWAP trend following. Buy when
price has stretched meaningfully below session VWAP (oversold relative to
the day's volume-weighted average), sell when it reverts back up to VWAP.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import session_vwap
from backtester.strategy import Bar, Lookback, Signal, Strategy


class VwapMeanReversionStrategy(Strategy):
    def __init__(self, min_bars: int = 5, entry_deviation_pct: float = 0.5, acceptance_bars: int = 1):
        self.min_bars = min_bars
        self.entry_deviation_pct = entry_deviation_pct  # % below VWAP that triggers a buy
        # How many CONSECUTIVE bars must stay beyond entry_deviation_pct before
        # entry fires ("acceptance confirmation" — source: Chris Drysdale/VWAP
        # Wave System video review, 2026-07-30; see CLAUDE_NOTES.txt PENDING
        # IDEAS). 1 (the default) is the ORIGINAL, unchanged behavior: fire on
        # the very first bar that crosses the threshold. >1 waits for the
        # deviation to be sustained ("accepted"), not just touched once, before
        # treating it as a real entry. This is an entry-condition EXPERIMENT on
        # the existing strategy, not a new one — don't change the
        # STRATEGY_REGISTRY default unless a walk-forward pass validates it
        # beats 1, same discipline as the Phase E param sweep.
        self.acceptance_bars = acceptance_bars

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

        deviation_pct = (closes - vwap) / vwap * 100
        oversold = deviation_pct <= -self.entry_deviation_pct

        # Fire once, exactly when a run of `acceptance_bars` consecutive
        # oversold bars is first completed — not on every bar of an even
        # longer run. acceptance_bars=1 reduces to "prev bar wasn't oversold,
        # current bar is", identical to this strategy's original behavior.
        if len(oversold) > self.acceptance_bars:
            recent_run = oversold.iloc[-self.acceptance_bars :]
            bar_before_run = oversold.iloc[-self.acceptance_bars - 1]
            entered_oversold = bool(recent_run.all()) and not bool(bar_before_run)
        else:
            entered_oversold = False
        # reverted back up to (or through) VWAP -> take profit
        reverted_to_vwap = prev_close < prev_vwap and curr_close >= curr_vwap

        if entered_oversold:
            return Signal.BUY
        if reverted_to_vwap:
            return Signal.SELL
        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far past the entry_deviation_pct trigger the current deviation
        below VWAP actually is, as a multiple of the threshold itself: 0.0 right
        at the threshold (barely triggered), 1.0 at double the threshold depth
        (clamped beyond that by compute_conviction). Added 2026-08-02 after
        trade-level attribution analysis showed this strategy's conviction score
        was nearly non-predictive (flat win rate across terciles) while using
        the generic fallback — unlike Bollinger Mean Reversion's own custom
        distance-past-threshold hook, which IS predictive (win rate and avg
        return both rise meaningfully in the top tercile). Mirrors that same
        "distance past threshold" idea, just scaled by this strategy's own fixed
        entry_deviation_pct instead of a rolling std, since VWAP MR's trigger is
        itself a fixed % distance, not a volatility band. (#25 — logged only,
        does not affect sizing.)
        """
        if len(history) < self.min_bars + 1:
            return None
        vwap = session_vwap(history)
        curr_vwap = vwap.iloc[-1]
        if pd.isna(curr_vwap) or self.entry_deviation_pct <= 0:
            return None
        curr_close = history["close"].iloc[-1]
        curr_deviation_pct = (curr_close - curr_vwap) / curr_vwap * 100
        if curr_deviation_pct >= 0:
            return None  # not actually below VWAP at all right now
        return (-curr_deviation_pct - self.entry_deviation_pct) / self.entry_deviation_pct

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        """Session VWAP, this bar's %-deviation from it, and the entry threshold
        - what this strategy is actually watching, for display (#detail request)."""
        if len(history) < self.min_bars + 1:
            return None
        vwap = session_vwap(history)
        curr_vwap = vwap.iloc[-1]
        if pd.isna(curr_vwap):
            return None
        curr_close = history["close"].iloc[-1]
        deviation_pct = (curr_close - curr_vwap) / curr_vwap * 100
        return {
            "vwap": float(curr_vwap),
            "deviation_pct": float(deviation_pct),
            "entry_threshold_pct": -self.entry_deviation_pct,
        }
