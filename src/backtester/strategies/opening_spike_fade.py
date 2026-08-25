"""Opening Spike Fade: fades whatever directional move happens in the first
`opening_minutes` of the session, targeting a reversion back to the session
open within `reversal_window_minutes`. Models the user's own observed
pattern (2026-08-23): the Nasdaq 100 often spikes hard in one direction
right at the open, then turns and heads the other way within roughly the
first hour.

Deliberately simpler than opening_range_liquidity_reversal.py - no
candlestick confirmation, no "oversized vs trailing average" filter, just
"did the opening window move more than min_move_pct, then fade it" -
testing the user's own, more direct description of the pattern rather than
assuming that strategy's extra criteria apply here too.

Bidirectional from the start (unlike opening_range_liquidity_reversal.py,
which is long-only "since this app has no shorting anywhere" as of when it
was written) - the engine's short-selling support (PositionMode,
2026-08-08, see engine.py) makes fading an UP spike by going SHORT just as
direct as fading a DOWN spike by going LONG.

Enters on exactly the FIRST bar after the opening window closes - one
attempt per session, matching "the open has ONE spike to fade", not a
repeated signal re-checked all through the reversal window. Exits on
whichever comes first: price reverting all the way back to the session
open (the natural full-reversion target), or the reversal-window time-stop.
The min_move_pct gate wraps BOTH the entry and exit checks together (not
just the entry) - a move too small to count as a "spike" must never enter,
which also means it must never emit an exit signal either, since an exit
signal on a day nothing was entered would be misread by the engine as a
fresh entry instead (shares == 0, so a stray SELL/BUY "exit" opens a new
position rather than closing one that was never there).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import session_dates
from backtester.strategy import Bar, Lookback, Signal, Strategy


class OpeningSpikeFadeStrategy(Strategy):
    def __init__(
        self,
        opening_minutes: int = 15,
        reversal_window_minutes: int = 60,
        min_move_pct: float = 0.1,
        max_move_pct: float | None = None,
    ):
        self.opening_minutes = opening_minutes
        self.reversal_window_minutes = reversal_window_minutes
        self.min_move_pct = min_move_pct
        # None = no cap (matches original behavior exactly). Added 2026-08-24
        # after per-session trade analysis found the LARGEST opening moves are
        # actually the LEAST reliable to fade (lowest win rate of any move-size
        # quartile, 62.9% vs 76.4% for the smallest) - the naive assumption
        # that a bigger spike is a better spike to fade turned out backwards.
        self.max_move_pct = max_move_pct

    def required_lookback(self) -> Lookback:
        return Lookback(sessions=1)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        dates = session_dates(history.index)
        today = dates.iloc[-1]
        today_bars = history[dates == today]
        if today_bars.empty:
            return Signal.HOLD

        session_start = today_bars.index[0]
        session_open = today_bars["open"].iloc[0]
        or_end = session_start + pd.Timedelta(minutes=self.opening_minutes)
        reversal_end = session_start + pd.Timedelta(minutes=self.reversal_window_minutes)

        if current.timestamp < or_end:
            return Signal.HOLD  # opening window still forming

        opening_window = today_bars[today_bars.index < or_end]
        if opening_window.empty or session_open <= 0:
            return Signal.HOLD

        opening_move_pct = (opening_window["close"].iloc[-1] - session_open) / session_open * 100
        abs_move = abs(opening_move_pct)
        if abs_move < self.min_move_pct:
            return Signal.HOLD  # too small to call a "spike" - no entry, no exit, today
        if self.max_move_pct is not None and abs_move > self.max_move_pct:
            return Signal.HOLD  # too large - the least reliable bucket to fade, see max_move_pct docstring

        faded_direction_long = opening_move_pct < 0  # spiked down -> fade by going long

        post_or_bars = today_bars[today_bars.index >= or_end]
        is_entry_bar = len(post_or_bars) == 1

        if is_entry_bar:
            return Signal.BUY if faded_direction_long else Signal.SELL

        reverted = (
            current.close >= session_open if faded_direction_long else current.close <= session_open
        )
        timed_out = current.timestamp >= reversal_end
        if reverted or timed_out:
            return Signal.SELL if faded_direction_long else Signal.BUY

        return Signal.HOLD
