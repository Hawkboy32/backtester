"""Flag/pennant continuation: a strong directional move (the "flagpole")
followed by a tight, low-volatility consolidation (the "flag"), then a
breakout in the original direction. There's no single precise algorithmic
definition of this chart pattern — this is a specific, parameterized
approximation:

1. Flagpole: over `flagpole_window` bars, close moved more than
   `min_flagpole_move_pct`.
2. Flag: over the following `consolidation_window` bars, the trading range
   contracted to less than `max_consolidation_range_pct` of the flagpole's
   own range.
3. Breakout: current close breaks above the consolidation range -> buy
   (continuation). Breaks below -> sell (setup failed).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class FlagPennantContinuationStrategy(Strategy):
    def __init__(
        self,
        flagpole_window: int = 15,
        consolidation_window: int = 8,
        min_flagpole_move_pct: float = 3.0,
        max_consolidation_range_pct: float = 40.0,
    ):
        self.flagpole_window = flagpole_window
        self.consolidation_window = consolidation_window
        self.min_flagpole_move_pct = min_flagpole_move_pct
        self.max_consolidation_range_pct = max_consolidation_range_pct

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.flagpole_window + self.consolidation_window + 2)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = self.flagpole_window + self.consolidation_window + 2
        if len(history) < min_len:
            return Signal.HOLD

        consolidation = history.iloc[-(self.consolidation_window + 1) : -1]
        flagpole = history.iloc[
            -(self.flagpole_window + self.consolidation_window + 1) : -(self.consolidation_window + 1)
        ]
        if flagpole.empty or consolidation.empty:
            return Signal.HOLD

        flagpole_move_pct = (
            (flagpole["close"].iloc[-1] - flagpole["close"].iloc[0]) / flagpole["close"].iloc[0] * 100
        )
        flagpole_range = flagpole["high"].max() - flagpole["low"].min()
        consolidation_range = consolidation["high"].max() - consolidation["low"].min()

        if flagpole_range <= 0:
            return Signal.HOLD

        consolidation_range_pct_of_pole = consolidation_range / flagpole_range * 100
        is_valid_setup = (
            flagpole_move_pct > self.min_flagpole_move_pct
            and consolidation_range_pct_of_pole < self.max_consolidation_range_pct
        )
        if not is_valid_setup:
            return Signal.HOLD

        consolidation_high = consolidation["high"].max()
        consolidation_low = consolidation["low"].min()

        if current.close > consolidation_high:
            return Signal.BUY
        if current.close < consolidation_low:
            return Signal.SELL
        return Signal.HOLD
