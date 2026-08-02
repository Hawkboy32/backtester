"""Opening Range Liquidity Reversal: fade an oversized ("liquidity" / stop-
hunt) opening move once it shows a reversal candlestick, targeting a return
to the other side of the opening range.

Source: independently described by three unrelated YouTube trading channels
(reviewed 2026-07-30/31), converging on the same mechanics — box the first N
minutes of the session, confirm the move is unusually large relative to
"normal" (their common reference is a 14-day ATR on DAILY bars — a separate
daily-bar fetch per ticker, mirroring volatility.py's GARCH plumbing, would
be needed to replicate that exactly; deliberately NOT built for this first
pass, to avoid new engine-level plumbing for one strategy), then wait for a
hammer/engulfing reversal candle and enter on the break of its high.

This strategy substitutes a SELF-CONTAINED proxy for the ATR filter instead:
today's opening-range size vs. the trailing average opening-range size over
`lookback_sessions` prior sessions — same underlying question ("is today's
opening move unusually big?"), computed entirely from the same intraday bars
the strategy already receives, no extra fetch. `lookback_sessions` defaults
to 5 (a trading week), not the videos' 14 days — this project's typical scan
window is only ~33 days, so a 14-session baseline would burn through nearly
half of it before the strategy could ever fire (same reasoning GARCH's
MIN_TRAIN/REGIME_MIN_PERIODS were shrunk from their vendored defaults, see
CLAUDE_NOTES.txt).

Long-only, matching every other strategy in this engine: only fades a
DOWNWARD liquidity candle (buying the reversal). The mirror-image short
setup (fading an upward liquidity candle) is described in the source videos
but deliberately not implemented — this app has no shorting anywhere.

Exit is checked FIRST and unconditionally, independent of all the entry-side
gating (liquidity-candle size, direction, the entry time window) — so a bar
where those conditions no longer hold (e.g. the reversal window has closed
for the day) can still close an already-open position. Exit fires on either
price reaching back to the opening-range high (target) or a fresh post-
opening-range low (the reversal thesis failed).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import is_bullish_engulfing, is_hammer, session_dates
from backtester.strategy import Bar, Lookback, Signal, Strategy


class OpeningRangeLiquidityReversalStrategy(Strategy):
    def __init__(
        self,
        opening_minutes: int = 15,
        reversal_window_minutes: int = 90,
        liquidity_multiplier: float = 1.5,
        lookback_sessions: int = 5,
    ):
        self.opening_minutes = opening_minutes
        self.reversal_window_minutes = reversal_window_minutes
        self.liquidity_multiplier = liquidity_multiplier
        self.lookback_sessions = lookback_sessions

    def required_lookback(self) -> Lookback:
        return Lookback(sessions=self.lookback_sessions + 1)

    def _opening_window(self, day_bars: pd.DataFrame) -> pd.DataFrame:
        session_start = day_bars.index[0]
        window_end = session_start + pd.Timedelta(minutes=self.opening_minutes)
        return day_bars[day_bars.index < window_end]

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < 3:
            return Signal.HOLD

        dates = session_dates(history.index)
        today = dates.iloc[-1]
        today_bars = history[dates == today]
        session_start = today_bars.index[0]
        or_end = session_start + pd.Timedelta(minutes=self.opening_minutes)

        if current.timestamp < or_end:
            return Signal.HOLD  # today's opening range is still forming

        opening_window = self._opening_window(today_bars)
        if opening_window.empty:
            return Signal.HOLD
        or_high = opening_window["high"].max()
        or_low = opening_window["low"].min()
        if or_high - or_low <= 0:
            return Signal.HOLD

        post_or_bars = today_bars[today_bars.index >= or_end]

        # --- exit: always evaluated, never gated behind the entry-side
        # checks below, so an open position can't get stuck once those stop
        # applying (e.g. the reversal window closes while still holding).
        reached_target = current.close >= or_high
        if len(post_or_bars) > 1:
            prior_post_or_low = post_or_bars["low"].iloc[:-1].min()
            made_fresh_low = current.low < prior_post_or_low
        else:
            made_fresh_low = False  # nothing prior in the post-OR window to compare against yet
        if reached_target or made_fresh_low:
            return Signal.SELL

        # --- entry ---
        reversal_end = session_start + pd.Timedelta(minutes=self.reversal_window_minutes)
        if current.timestamp > reversal_end:
            return Signal.HOLD  # past today's entry window

        prior_dates = pd.unique(dates[dates != today])
        if len(prior_dates) < self.lookback_sessions:
            return Signal.HOLD  # not enough trailing sessions yet to judge "oversized"

        prior_ranges = []
        for d in prior_dates[-self.lookback_sessions :]:
            day_window = self._opening_window(history[dates == d])
            if not day_window.empty:
                prior_ranges.append(day_window["high"].max() - day_window["low"].min())
        if not prior_ranges:
            return Signal.HOLD
        avg_prior_range = sum(prior_ranges) / len(prior_ranges)
        if avg_prior_range <= 0 or (or_high - or_low) < self.liquidity_multiplier * avg_prior_range:
            return Signal.HOLD  # today's opening move isn't oversized enough to count

        if opening_window["close"].iloc[-1] >= opening_window["open"].iloc[0]:
            return Signal.HOLD  # only fade DOWNWARD liquidity candles (long-only)

        candidate = history.iloc[-2]
        if not (or_end <= candidate.name <= reversal_end):
            return Signal.HOLD
        if candidate["low"] > or_low:
            return Signal.HOLD  # candidate never tested/broke the opening-range low

        prior_candidate = history.iloc[-3]
        reversal_confirmed = is_hammer(
            candidate["open"], candidate["high"], candidate["low"], candidate["close"]
        ) or is_bullish_engulfing(
            prior_candidate["open"], prior_candidate["close"], candidate["open"], candidate["close"]
        )
        if not reversal_confirmed:
            return Signal.HOLD

        if current.close > candidate["high"]:
            return Signal.BUY
        return Signal.HOLD
