"""Opening Range Breakout (ORB): lock in the high/low of the first N minutes
of each trading session, then buy when price breaks above that range.
Exits on a break back below the range low (stop) — deliberately not an
"exit at end of day" rule, since that would need future bars a live
strategy can't see.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import session_dates
from backtester.strategy import Bar, Lookback, Signal, Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    def __init__(self, opening_minutes: int = 15):
        self.opening_minutes = opening_minutes

    def required_lookback(self) -> Lookback:
        return Lookback(sessions=1)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < 2:
            return Signal.HOLD

        dates = session_dates(history.index)
        today = dates.iloc[-1]
        today_bars = history[dates == today]
        session_start = today_bars.index[0]
        window_end = session_start + pd.Timedelta(minutes=self.opening_minutes)

        if current.timestamp < window_end:
            return Signal.HOLD  # today's opening range is still forming

        opening_window = today_bars[today_bars.index < window_end]
        if opening_window.empty:
            return Signal.HOLD
        or_high = opening_window["high"].max()
        or_low = opening_window["low"].min()

        post_window = today_bars[today_bars.index >= window_end]
        if len(post_window) < 2:
            # first bar after the opening range closed — no prior post-window
            # bar to detect a crossing against, just check an outright breakout
            if current.close > or_high:
                return Signal.BUY
            return Signal.HOLD

        prev_close = post_window["close"].iloc[-2]
        curr_close = current.close

        crossed_up = prev_close <= or_high and curr_close > or_high
        crossed_down = prev_close >= or_low and curr_close < or_low

        if crossed_up:
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD
