"""Detrended Price Oscillator mean reversion: buy when DPO drops into its own
oversold band, sell on reversion back to zero (DPO's own "detrended mean").

Sourced from AlphaInsider strategy-browsing (2026-09-06) as a candidate worth
testing - DPO (Investopedia/StockCharts standard formula: a past close minus
the SMA ending today, see indicators.detrended_price_oscillator's own
docstring) isolates cycles by removing trend, distinct from RSI's momentum-
based approach even though both end up as "buy the dip" oscillators.

The oversold band is DPO's own rolling std-dev (not a fixed raw-price-scale
number, which wouldn't generalize across tickers) - same num_std convention
Bollinger Mean Reversion already uses, applied to DPO instead of price.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategies.indicators import detrended_price_oscillator
from backtester.strategy import Bar, Lookback, Signal, Strategy


class DpoMeanReversionStrategy(Strategy):
    def __init__(self, period: int = 20, num_std: float = 1.5):
        self.period = period
        self.num_std = num_std

    def required_lookback(self) -> Lookback:
        # DPO itself needs `period` bars for its SMA, PLUS shifts a close
        # back by period//2+1 more - both baked into detrended_price_oscillator.
        shift = self.period // 2 + 1
        return Lookback(bars=self.period * 2 + shift + 2)

    def _dpo_and_band(self, history: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        dpo = detrended_price_oscillator(history["close"], self.period)
        band = self.num_std * dpo.rolling(self.period).std()
        return dpo, band

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        min_len = self.period * 2 + (self.period // 2 + 1)
        if len(history) < min_len:
            return Signal.HOLD

        dpo, band = self._dpo_and_band(history)
        if len(dpo) < 2 or pd.isna(dpo.iloc[-2]) or pd.isna(band.iloc[-1]):
            return Signal.HOLD

        prev_dpo, curr_dpo = dpo.iloc[-2], dpo.iloc[-1]
        prev_band, curr_band = -band.iloc[-2], -band.iloc[-1]

        broke_down = prev_dpo >= prev_band and curr_dpo < curr_band
        reverted_to_zero = curr_dpo > 0

        if broke_down:
            return Signal.BUY
        if reverted_to_zero:
            return Signal.SELL
        return Signal.HOLD

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """How far below its own oversold band DPO sits, as a fraction of the
        band width (#25 - logged only, does not affect sizing)."""
        min_len = self.period * 2 + (self.period // 2 + 1)
        if len(history) < min_len:
            return None
        dpo, band = self._dpo_and_band(history)
        curr_dpo, curr_band = dpo.iloc[-1], band.iloc[-1]
        if pd.isna(curr_dpo) or pd.isna(curr_band) or curr_band <= 0:
            return None
        return (-curr_band - curr_dpo) / curr_band

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        min_len = self.period * 2 + (self.period // 2 + 1)
        if len(history) < min_len:
            return None
        dpo, band = self._dpo_and_band(history)
        curr_dpo, curr_band = dpo.iloc[-1], band.iloc[-1]
        if pd.isna(curr_dpo) or pd.isna(curr_band):
            return None
        return {"dpo": float(curr_dpo), "oversold": float(-curr_band), "overbought": float(curr_band)}
