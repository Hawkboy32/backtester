"""VWAP drift pullback: trend-CONTINUATION off the session VWAP, the
opposite bet from VwapMeanReversionStrategy despite sharing an indicator.

Idea (own adaptation of a "VWAP + pullback + momentum filter" day-trading
concept reviewed 2026-08-10, re-derived for percentage/bar terms rather than
the original's fixed-point futures terms — see CLAUDE_NOTES.txt "PENDING
IDEAS" for the source and how this differs from it, notably: this project's
engine is signal-driven with no per-trade stop/target order support, so the
exit here is a VWAP-cross invalidation rather than a fixed reward:risk
bracket, and every threshold below is a fresh parameter to actually validate
on OUR instruments, not copied from a system built for point-based futures
and a different objective (passing a firm's evaluation window, not growing
an account) that its own creator says isn't meant to trade real capital as-is):

A session VWAP that is itself SLOPING in one direction, combined with recent
price momentum agreeing with that slope, is read as "there's an established
drift today, not just noise." Once both hold, the entry trigger is the
first pullback candle back toward VWAP — buying the first red candle in an
uptrend, selling the first green candle in a downtrend — on the idea that a
trending VWAP acts as a magnet for the execution flow still working an order
in that direction, not a level price reverses hard away from (the opposite
read from mean reversion's "stretched too far, snap back").
"""

from __future__ import annotations

from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

from backtester.strategies.indicators import session_vwap
from backtester.strategy import Bar, Lookback, Signal, Strategy

# Bars come in UTC-indexed (see data.PolygonClient.get_aggregates) and, for
# equities, include pre/after-market — so "skip the first N bars of the
# session" is not the same as "skip the first N minutes of REGULAR trading"
# unless bars are first filtered down to the regular session by wall-clock
# time. Converting via ZoneInfo (not a fixed UTC offset) is what makes this
# correctly DST-aware, same approach as brokers.ibkr._us_equity_clock_heuristic.
_ET = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)


class VwapDriftPullbackStrategy(Strategy):
    def __init__(
        self,
        min_bars: int = 5,
        vwap_slope_lookback_bars: int = 15,
        momentum_lookback_bars: int = 60,
        momentum_threshold_pct: float = 0.1,
        skip_opening_minutes: int = 60,
    ):
        self.min_bars = min_bars
        self.vwap_slope_lookback_bars = vwap_slope_lookback_bars
        self.momentum_lookback_bars = momentum_lookback_bars
        self.momentum_threshold_pct = momentum_threshold_pct
        # Skips the first stretch of the REGULAR session (by wall-clock time,
        # not bar count) so VWAP has enough volume behind it to mean
        # something — an early-session VWAP is dominated by whatever the
        # opening print happened to be, not a real volume-weighted consensus
        # yet. Also naturally excludes pre-market entirely.
        self.skip_opening_minutes = skip_opening_minutes

    def required_lookback(self) -> Lookback:
        # sessions=2 for the same reason as VwapMeanReversionStrategy/
        # VwapTrendStrategy (the day-boundary VWAP slot), plus enough bars for
        # the longer of the two lookback windows. Extra headroom since
        # regular-hours bars are now a subset of what's fetched (extended
        # hours bars still count toward Lookback.bars).
        needed = max(self.vwap_slope_lookback_bars, self.momentum_lookback_bars) + 2
        return Lookback(sessions=2, bars=needed * 2)

    def _in_tradeable_window(self, current: Bar) -> bool:
        """True once wall-clock time (America/New_York, DST-aware) is past
        the regular-session open plus the opening skip, and before close —
        i.e. excludes pre-market, the opening skip, and after-hours alike."""
        local = current.timestamp.tz_convert(_ET)
        if local.weekday() >= 5:
            return False
        local_time = local.time()
        session_start = (
            pd.Timestamp.combine(local.date(), _REGULAR_OPEN) + pd.Timedelta(minutes=self.skip_opening_minutes)
        ).time()
        return session_start <= local_time < _REGULAR_CLOSE

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        needed = max(self.vwap_slope_lookback_bars, self.momentum_lookback_bars) + 1
        if len(history) < max(self.min_bars, needed) + 1:
            return Signal.HOLD
        if not self._in_tradeable_window(current):
            return Signal.HOLD

        vwap = session_vwap(history)
        closes = history["close"]
        opens = history["open"]

        curr_vwap = vwap.iloc[-1]
        slope_vwap = vwap.iloc[-1 - self.vwap_slope_lookback_bars]
        momentum_base = closes.iloc[-1 - self.momentum_lookback_bars]
        if pd.isna(curr_vwap) or pd.isna(slope_vwap) or pd.isna(momentum_base) or momentum_base == 0:
            return Signal.HOLD

        curr_close, curr_open = closes.iloc[-1], opens.iloc[-1]
        momentum_pct = (curr_close - momentum_base) / momentum_base * 100

        above_vwap = curr_close > curr_vwap
        below_vwap = curr_close < curr_vwap
        vwap_rising = curr_vwap > slope_vwap
        vwap_falling = curr_vwap < slope_vwap
        pullback_candle_down = curr_close < curr_open  # red candle -> long trigger
        pullback_candle_up = curr_close > curr_open  # green candle -> short trigger

        drift_up = above_vwap and vwap_rising and momentum_pct >= self.momentum_threshold_pct
        drift_down = below_vwap and vwap_falling and momentum_pct <= -self.momentum_threshold_pct

        if drift_up and pullback_candle_down:
            return Signal.BUY
        if drift_down and pullback_candle_up:
            return Signal.SELL

        # Invalidation exit: the whole thesis was "price holds on the drift's
        # side of a trending VWAP" - once that flips, there's no edge left to
        # hold for, independent of which side opened the position.
        if above_vwap is False and below_vwap is False:
            pass  # exactly on VWAP - let the next bar resolve it
        elif below_vwap:
            return Signal.SELL  # closes a long that's now below VWAP
        elif above_vwap:
            return Signal.BUY  # closes a short that's now above VWAP

        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far past the momentum threshold the current push actually is,
        as a multiple of the threshold itself - same "distance past trigger"
        shape as VwapMeanReversionStrategy's own conviction hook, just scaled
        by momentum_threshold_pct instead of entry_deviation_pct."""
        needed = max(self.vwap_slope_lookback_bars, self.momentum_lookback_bars) + 1
        if len(history) < max(self.min_bars, needed) + 1 or self.momentum_threshold_pct <= 0:
            return None
        closes = history["close"]
        momentum_base = closes.iloc[-1 - self.momentum_lookback_bars]
        if pd.isna(momentum_base) or momentum_base == 0:
            return None
        momentum_pct = (closes.iloc[-1] - momentum_base) / momentum_base * 100
        return (abs(momentum_pct) - self.momentum_threshold_pct) / self.momentum_threshold_pct

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        needed = max(self.vwap_slope_lookback_bars, self.momentum_lookback_bars) + 1
        if len(history) < max(self.min_bars, needed) + 1:
            return None
        vwap = session_vwap(history)
        curr_vwap = vwap.iloc[-1]
        slope_vwap = vwap.iloc[-1 - self.vwap_slope_lookback_bars]
        if pd.isna(curr_vwap) or pd.isna(slope_vwap):
            return None
        closes = history["close"]
        momentum_base = closes.iloc[-1 - self.momentum_lookback_bars]
        momentum_pct = (
            float((closes.iloc[-1] - momentum_base) / momentum_base * 100)
            if not pd.isna(momentum_base) and momentum_base != 0
            else 0.0
        )
        return {
            "vwap": float(curr_vwap),
            "vwap_slope": float(curr_vwap - slope_vwap),
            "momentum_pct": momentum_pct,
        }
