"""Per-signal conviction scoring for the measure-first learning dataset (#25).

Conviction is a [0, 1] strength score for an entry signal — 0 = barely
triggered, 1 = strong. It is **logged only**: nothing in this pass sizes or
gates a trade on it (measure-first, per the project's family-wealth caution
posture). It exists to build the dataset a future conviction-sizing / ML
meta-model will learn from.

Two sources, unified by compute_conviction():
- a strategy's own `conviction()` hook (strength specific to its setup), or
- a generic fallback from entry-bar context when a strategy doesn't provide one,
  so EVERY logged trade carries a conviction rather than a hole.

FUTURE (#21): once the daily-news panel exists, per-ticker news sentiment is
intended as an additional input here — blend it into compute_conviction then.
"""

from __future__ import annotations

import pandas as pd

from backtester.strategy import Bar, Strategy


def clamp01(x: float | None) -> float:
    """Coerce to [0, 1]; NaN/None -> 0.0."""
    if x is None or x != x:  # None or NaN
        return 0.0
    return max(0.0, min(1.0, float(x)))


def compute_conviction(strategy: Strategy, history: pd.DataFrame, current: Bar) -> float:
    """Return a [0, 1] conviction for the entry at `current`. Uses the strategy's
    own conviction() hook when it returns a value; otherwise the generic fallback.
    A buggy hook never propagates — it just falls through to the fallback."""
    try:
        value = strategy.conviction(history, current)
    except Exception:  # noqa: BLE001 — a strategy's optional hook must never break a trade
        value = None
    if value is not None:
        return clamp01(value)
    return _generic_conviction(history, current)


def compute_levels(strategy: Strategy, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
    """Return the strategy's current reference levels (see Strategy.levels), or
    None if it doesn't define any. DISPLAY ONLY. A buggy hook never propagates —
    same fail-safe pattern as compute_conviction."""
    try:
        return strategy.levels(history, current)
    except Exception:  # noqa: BLE001 — an optional display hook must never break a trade
        return None


def _generic_conviction(history: pd.DataFrame, current: Bar, lookback: int = 20) -> float:
    """Fallback strength from entry-bar context, for strategies without a bespoke
    hook. v1 heuristic (NOT ground truth): blend (a) the size of the current bar's
    move vs recent volatility and (b) volume vs its recent average, squashed to
    [0, 1]. A rough proxy — it's data to learn from later, not a trade signal now.
    Returns a neutral 0.5 when there isn't enough context to judge."""
    closes = history["close"]
    if len(closes) < 3:
        return 0.5

    # (a) volatility-normalised move: |this bar's return| / recent std of returns.
    returns = closes.pct_change().dropna()
    if returns.empty:
        return 0.5
    recent = returns.tail(lookback)
    vol = float(recent.std()) if len(recent) >= 2 else 0.0
    last_ret = float(returns.iloc[-1])
    move_z = abs(last_ret) / vol if vol > 1e-9 else 0.0
    move_score = clamp01(move_z / 3.0)  # ~3-sigma move -> 1.0

    # (b) relative volume vs its recent average.
    vol_score = 0.5
    if "volume" in history.columns and len(history) >= 3:
        vols = history["volume"].tail(lookback)
        avg_v = float(vols.mean())
        if avg_v > 0 and current.volume is not None:
            vol_score = clamp01((current.volume / avg_v) / 3.0)  # 3x average volume -> 1.0

    return clamp01(0.6 * move_score + 0.4 * vol_score)
