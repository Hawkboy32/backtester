"""DMI (+DI/-DI only, no ADX) gated by DPO as a noise guard: buy when +DI
crosses above -DI while DPO confirms a real (non-flat) move is underway,
sell on the opposite DI crossover.

Sourced from AlphaInsider strategy-browsing (2026-09-27) as a candidate worth
testing - the author's own description explains the reasoning directly:
DMI's usual ADX filter throws false trend signals inside a "trade range
zone" (chop), so this drops ADX entirely and uses DPO instead to suppress
entries when the market is flat. Genuinely different from this project's
existing DmiAdxTrendStrategy (uses ADX to gate) and DpoMeanReversionStrategy
(DPO drives entries directly, not just a filter) - this combines the two
building blocks in a way neither does alone.

The "is this flat/noise" test reuses the exact convention
DpoMeanReversionStrategy already established: DPO's own rolling std-dev, not
a fixed raw-price-scale number, so it generalizes across tickers/vol regimes.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import detrended_price_oscillator, directional_movement_index
from backtester.strategy import Bar, Lookback, Signal, Strategy


class DmiDpoGuardStrategy(Strategy):
    def __init__(self, dmi_period: int = 14, dpo_period: int = 20, dpo_guard_std: float = 0.5):
        self.dmi_period = dmi_period
        self.dpo_period = dpo_period
        # How far outside its own noise band |DPO| must sit before a DI
        # crossover is trusted. 0.5, not DPO Mean-Reversion's 1.5 - that
        # strategy needs an extreme DPO reading to trade DPO itself; this one
        # only needs DPO to confirm "not flat", a much lower bar.
        self.dpo_guard_std = dpo_guard_std

    def required_lookback(self) -> Lookback:
        dpo_shift = self.dpo_period // 2 + 1
        dmi_bars = self.dmi_period * 6  # same generous EWM-warmup multiple DmiAdxTrendStrategy uses
        dpo_bars = self.dpo_period * 2 + dpo_shift + 2
        return Lookback(bars=max(dmi_bars, dpo_bars))

    def _dpo_guard_passes(self, history: pd.DataFrame) -> bool:
        dpo = detrended_price_oscillator(history["close"], self.dpo_period)
        band = self.dpo_guard_std * dpo.rolling(self.dpo_period).std()
        if pd.isna(dpo.iloc[-1]) or pd.isna(band.iloc[-1]):
            return False
        return abs(dpo.iloc[-1]) > band.iloc[-1]

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = min(self.dmi_period * 3, self.dpo_period * 2 + (self.dpo_period // 2 + 1))
        if len(history) < min_len:
            return Signal.HOLD

        plus_di, minus_di, _ = directional_movement_index(history, self.dmi_period)
        if len(plus_di) < 2 or pd.isna(plus_di.iloc[-2]) or pd.isna(minus_di.iloc[-1]):
            return Signal.HOLD

        prev_plus, curr_plus = plus_di.iloc[-2], plus_di.iloc[-1]
        prev_minus, curr_minus = minus_di.iloc[-2], minus_di.iloc[-1]

        crossed_up = prev_plus <= prev_minus and curr_plus > curr_minus
        crossed_down = prev_plus >= prev_minus and curr_plus < curr_minus

        # The guard only gates NEW entries, same principle DmiAdxTrendStrategy
        # already uses for its ADX filter - an open position can still be
        # exited on the opposite crossover even if DPO has since flattened.
        if crossed_up and self._dpo_guard_passes(history):
            return Signal.BUY
        if crossed_down:
            return Signal.SELL
        return Signal.HOLD

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        min_len = min(self.dmi_period * 3, self.dpo_period * 2 + (self.dpo_period // 2 + 1))
        if len(history) < min_len:
            return None
        plus_di, minus_di, _ = directional_movement_index(history, self.dmi_period)
        dpo = detrended_price_oscillator(history["close"], self.dpo_period)
        band = self.dpo_guard_std * dpo.rolling(self.dpo_period).std()
        curr_plus, curr_minus = plus_di.iloc[-1], minus_di.iloc[-1]
        curr_dpo, curr_band = dpo.iloc[-1], band.iloc[-1]
        if pd.isna(curr_plus) or pd.isna(curr_minus):
            return None
        return {
            "plus_di": float(curr_plus),
            "minus_di": float(curr_minus),
            "dpo": float(curr_dpo) if not pd.isna(curr_dpo) else 0.0,
            "dpo_guard_band": float(curr_band) if not pd.isna(curr_band) else 0.0,
        }
