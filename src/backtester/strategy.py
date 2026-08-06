"""Strategy interface for the backtest engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Signal(Enum):
    HOLD = "hold"
    BUY = "buy"
    SELL = "sell"


@dataclass
class Bar:
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Lookback:
    """How much trailing history a strategy actually needs to produce a
    correct signal at the current bar — lets the engine hand it a bounded
    window instead of the full history-so-far (which would otherwise force
    every rolling/EMA/session calculation to be recomputed from scratch over
    an ever-growing slice every single bar).

    Set exactly one of the two, matching whichever kind of dependency the
    strategy actually has:
    - `bars`: a fixed trailing bar count (rolling windows, EMAs, shift()).
    - `sessions`: a fixed trailing *session* (calendar day) count, for
      anything that resets daily or looks up a value keyed by session date
      (VWAP, opening range, prior-session pivot points) — bar-count varies
      with timespan/session length, so this is expressed in sessions instead.

    Leaving both None (the default) means "unbounded" — the engine falls
    back to passing the full history-so-far, exactly as before. This is the
    safe default for any strategy that hasn't been audited/opted in yet.
    """

    bars: int | None = None
    sessions: int | None = None


class Strategy(ABC):
    """Subclass this and implement on_bar to define a trading strategy.

    on_bar is called once per bar, in chronological order, with history
    up to and including the current bar — bounded to whatever
    required_lookback() declares is actually needed (unbounded/full history
    by default). Return a Signal.
    """

    @abstractmethod
    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        raise NotImplementedError

    def required_lookback(self) -> Lookback:
        """Override to opt into windowed (fast) backtesting by declaring how
        much trailing history this strategy actually needs. Unbounded by
        default (full history-so-far, same as before this existed).
        """
        return Lookback()

    def conviction(self, history: pd.DataFrame, current: Bar) -> float | None:
        """Optional: the strength of the entry signal at `current`, in [0, 1]
        (0 = weak/barely triggered, 1 = strong), or None to fall back to the
        generic context-based score (see backtester.conviction). Only consulted
        on the bar a BUY actually fires. LOGGED ONLY — never used to size or gate
        a trade in this measure-first pass. Default None; override where a
        natural strength exists (e.g. how far past a band/threshold price is).
        """
        return None

    def levels(self, history: pd.DataFrame, current: Bar) -> dict[str, float] | None:
        """Optional: the reference values (band/VWAP/threshold levels etc.) this
        strategy is currently watching, for display purposes only — e.g. so a
        human can see WHY a signal fired, not just that it did. Called every bar
        (not just on entry), unlike conviction(). DISPLAY ONLY — never consulted
        for sizing or gating. Default None; override where the strategy has
        natural reference levels worth surfacing (see backtester.conviction's
        compute_levels() for the safe wrapper callers should use)."""
        return None
