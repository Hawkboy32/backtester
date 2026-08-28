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


def turn_confirmed_uptick(closes: pd.Series, in_zone: pd.Series, confirm_bars: int) -> bool:
    """True if the last `confirm_bars` bars show a sustained uptick STARTING
    from a bar where `in_zone` (e.g. VWAP-mean-reversion's "oversold", or
    Bollinger's "below the lower band") was true — a "wait for the turn"
    entry confirmation, as opposed to firing the instant `in_zone` is first
    true. Added 2026-08-27/28 for VwapMeanReversionStrategy and
    BollingerMeanReversionStrategy's `confirm_turn_bars` param; shared here
    since both strategies need the identical shape, just against a
    different `in_zone` series.

    A PURE function of the passed-in Series - no internal state - because
    both callers are re-instantiated fresh every polling cycle in live
    trading (confirmed: auto_trader.py's _trade_target() calls
    build_strategy() every cycle), so nothing persisted on `self` between
    calls would ever survive to be useful there. `history` is always a
    pandas object anyway, so recomputing this from it each call costs
    nothing extra worth avoiding.

    Deliberately does NOT try to isolate the exact FIRST bar of a longer
    up-run (i.e., this can stay True for several consecutive bars once a
    qualifying run starts) - the caller's engine only acts on a BUY signal
    while flat, so a predicate held true for multiple bars in a row still
    only opens one position, on whichever bar it first goes true. Handling
    that inside this function would just be duplicating logic the engine
    already gets right.

    The baseline bar (`confirm_bars` back from the end) must itself be
    `in_zone` - the run has to start FROM the zone that would otherwise have
    triggered an instant entry, not from some already-recovered point partway
    through the reversion.
    """
    n = len(closes)
    if confirm_bars <= 0 or n <= confirm_bars:
        return False
    baseline_idx = n - 1 - confirm_bars
    if not bool(in_zone.iloc[baseline_idx]):
        return False
    recent = closes.iloc[baseline_idx:]
    return bool((recent.diff().iloc[1:] > 0).all())


def rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)
