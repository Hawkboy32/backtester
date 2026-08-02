"""Shared indicator primitives used by multiple strategies.

Kept here rather than duplicated per-strategy since several strategies need
the same underlying math (EMA, session boundaries, swing points).
"""

from __future__ import annotations

import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def ema_warmup_bars(span: int, tolerance_k: float = 20.0) -> int:
    """How many leading bars an ewm(span=..., adjust=False) recursion needs
    before its value is numerically converged to within roughly exp(-tolerance_k)
    relative error of the "true" (infinite-history) value.

    ewm(adjust=False) is a recursive filter: bar `k` steps back still
    contributes a (1-alpha)**k share of the current value, technically all
    the way back to bar zero. That share decays exponentially, though, so a
    long-enough trailing window is numerically indistinguishable from the
    unbounded original for any reasonable tolerance — this sizes that
    window instead of requiring the strategy's full history from bar zero.
    Default tolerance_k=20 means a relative error of roughly 2e-9, far
    tighter than anything that could matter for a trading decision.
    """
    alpha = 2.0 / (span + 1)
    return int(tolerance_k / alpha) + 1


def session_dates(index: pd.DatetimeIndex) -> pd.Series:
    """Calendar date (session) each bar belongs to — the grouping key for
    anything that resets daily (VWAP, opening range, pivot points).
    """
    return pd.Series(index.date, index=index)


def session_vwap(bars: pd.DataFrame) -> pd.Series:
    """Volume-weighted average price, cumulative within each trading day and
    resetting at the start of the next one — the standard definition, not a
    rolling window.
    """
    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    pv = typical_price * bars["volume"]
    dates = session_dates(bars.index)
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = bars["volume"].groupby(dates).cumsum()
    return cum_pv / cum_vol.replace(0, pd.NA)


def opening_range(bars: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Per-session high/low of the first `minutes` of each trading day.
    Returns a DataFrame indexed by session date with columns or_high/or_low.
    Assumes roughly minute-granularity bars; for coarser granularity the
    "opening range" degenerates to the first bar or two.
    """
    dates = session_dates(bars.index)
    grouped = bars.groupby(dates)

    records = []
    for date, group in grouped:
        session_start = group.index[0]
        window = group[group.index < session_start + pd.Timedelta(minutes=minutes)]
        if window.empty:
            window = group.iloc[:1]
        records.append({"date": date, "or_high": window["high"].max(), "or_low": window["low"].min()})

    return pd.DataFrame(records).set_index("date")


def prior_session_pivot_points(bars: pd.DataFrame) -> pd.DataFrame:
    """Standard (5-point) pivot points for each session, computed from the
    *previous* session's high/low/close: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H,
    R2 = P+(H-L), S2 = P-(H-L). Returns a DataFrame indexed by session date
    (the date the levels apply *to*, not the date they were computed from).
    """
    dates = session_dates(bars.index)
    daily = bars.groupby(dates).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))

    pivot = (daily["high"] + daily["low"] + daily["close"]) / 3
    r1 = 2 * pivot - daily["low"]
    s1 = 2 * pivot - daily["high"]
    r2 = pivot + (daily["high"] - daily["low"])
    s2 = pivot - (daily["high"] - daily["low"])

    levels = pd.DataFrame({"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2})
    return levels.shift(1)  # each day trades against the *prior* day's levels


def swing_high(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()


def swing_low(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()


def fibonacci_levels(high: float, low: float) -> dict[str, float]:
    """Standard retracement levels between a swing high and swing low."""
    diff = high - low
    return {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low,
    }


def is_hammer(open_: float, high: float, low: float, close: float, body_to_wick_ratio: float = 2.0) -> bool:
    """Classic bullish-reversal candlestick shape: a small body near the top
    of the bar's range with a lower wick at least `body_to_wick_ratio` times
    the body size and a negligible upper wick. Purely a shape check — that
    it actually follows a decline (the context that makes a hammer mean
    anything) is the caller's responsibility.
    """
    total_range = high - low
    if total_range <= 0:
        return False
    body = abs(close - open_)
    if body == 0:
        body = total_range * 0.001  # doji: still needs a real lower wick below, not a div-by-zero
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)
    return lower_wick >= body_to_wick_ratio * body and upper_wick <= body


def is_bullish_engulfing(prev_open: float, prev_close: float, curr_open: float, curr_close: float) -> bool:
    """Current candle is green, the previous one is red, and the current
    candle's body fully engulfs the previous candle's body."""
    prev_red = prev_close < prev_open
    curr_green = curr_close > curr_open
    engulfs = curr_open <= prev_close and curr_close >= prev_open
    return prev_red and curr_green and engulfs


def bearish_fvg_gap(low_two_bars_ago: float, high_current: float) -> float | None:
    """Bearish Fair Value Gap (the standard 3-candle imbalance concept):
    the gap between the LOW of the candle 2 bars back and the HIGH of the
    current candle, when price has moved down fast enough that the two
    don't overlap at all. The middle candle (1 bar back) doesn't factor
    into the definition itself — only whether these outer two touch.
    Returns the gap size (positive = a real, unfilled imbalance exists) or
    None when there's no gap (the ranges touch or overlap). Source: "Scalping
    Trading For Beginners" video review, 2026-07-30 — see CLAUDE_NOTES.txt
    PENDING IDEAS. The mirror-image bullish version (gap between the HIGH
    two bars back and the LOW of the current candle) isn't implemented —
    not needed by any strategy in this project yet (long-only, downward-
    move confirmation only), add it the same way if that changes.
    """
    gap = low_two_bars_ago - high_current
    return gap if gap > 0 else None


def rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)
