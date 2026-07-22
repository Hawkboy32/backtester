"""Pivot point scalping: trade bounces off standard pivot levels computed
from the *prior* session's high/low/close. Buy on a rejection off S1
(intrabar dip to S1 that closes back above it), take profit back at the
pivot level.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class PivotPointScalpingStrategy(Strategy):
    def required_lookback(self) -> Lookback:
        # today's session plus the one immediately prior — pivot levels are
        # shift(1) off the prior session's OHLC only, nothing further back.
        return Lookback(sessions=2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < 2:
            return Signal.HOLD

        # With the sessions=2 window, history holds exactly today plus the
        # prior session — so "the prior session's OHLC" is just every bar not
        # dated today. Same numbers prior_session_pivot_points() would give,
        # without re-aggregating every session on every bar.
        dates = history.index.date
        today = dates[-1]
        prior_bars = history[dates != today]

        if prior_bars.empty:
            return Signal.HOLD  # no prior session yet to compute levels from

        prior_high = prior_bars["high"].max()
        prior_low = prior_bars["low"].min()
        prior_close = prior_bars["close"].iloc[-1]

        pivot = (prior_high + prior_low + prior_close) / 3
        s1 = 2 * pivot - prior_high

        touched_support = current.low <= s1 and current.close > s1
        reached_pivot = current.close >= pivot

        if touched_support:
            return Signal.BUY
        if reached_pivot:
            return Signal.SELL
        return Signal.HOLD
