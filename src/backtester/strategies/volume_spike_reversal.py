"""Volume spike reversal (bullish variant, long-only): a bar with volume
well above its recent average, following a downward drift, that closes in
the upper portion of its own range — a "climax" bar suggesting sellers got
exhausted and buyers took control late in the bar. Exit when price starts
closing weaker again (momentum fading).
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Lookback, Signal, Strategy


class VolumeSpikeReversalStrategy(Strategy):
    def __init__(
        self,
        volume_window: int = 20,
        volume_multiple: float = 2.0,
        close_position_threshold: float = 0.6,
    ):
        self.volume_window = volume_window
        self.volume_multiple = volume_multiple
        self.close_position_threshold = close_position_threshold

    def required_lookback(self) -> Lookback:
        return Lookback(bars=self.volume_window + 6)

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        if len(history) < self.volume_window + 6:
            return Signal.HOLD

        avg_volume = history["volume"].iloc[:-1].rolling(self.volume_window).mean().iloc[-1]
        if pd.isna(avg_volume) or avg_volume <= 0:
            return Signal.HOLD

        volume_spike = current.volume > avg_volume * self.volume_multiple

        bar_range = current.high - current.low
        if bar_range <= 0:
            return Signal.HOLD
        close_position = (current.close - current.low) / bar_range  # 0 = closed at low, 1 = closed at high

        recent_trend_down = current.close < history["close"].iloc[-6:-1].mean()

        bullish_reversal = (
            volume_spike and close_position >= self.close_position_threshold and recent_trend_down
        )
        if bullish_reversal:
            return Signal.BUY

        fading = current.close < history["close"].iloc[-3:-1].mean()
        if fading:
            return Signal.SELL

        return Signal.HOLD
