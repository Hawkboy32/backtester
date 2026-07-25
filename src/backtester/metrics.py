"""Performance metrics computed from a backtest equity curve."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_TRADING_DAY = 390


@dataclass
class PerformanceReport:
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    num_trades: int
    win_rate: float
    # Per-trade expectancy ("casino maths"): what an average trade returns, as a
    # fraction of the capital committed to it. expectancy = win_rate * avg_win
    # + (1 - win_rate) * avg_loss (avg_loss is negative). Display/learning
    # metric only — deliberately NOT part of ranking's score weights.
    avg_win_pct: float | None = None
    avg_loss_pct: float | None = None
    expectancy: float | None = None
    # Profit factor: gross profit / gross loss (both positive). >1 is profitable.
    # More informative than win rate, which says nothing about SIZE — a strategy
    # winning 40% of the time is very profitable if the winners are big enough.
    # None when it is undefined: either no closed trades, or no losing trades at
    # all (a divide by zero, which on a small sample means "too few trades to
    # judge" far more often than it means "flawless").
    profit_factor: float | None = None
    # Fraction of the backtest's BARS with a position open. Counted in bars, not
    # wall-clock, so an overnight hold on minute data isn't scored as 16 hours of
    # exposure when no bars elapsed. Low time-in-market for the same return means
    # the same money was at risk for less of the window.
    time_in_market: float | None = None
    # Mean capital committed per trade (entry price x shares), in dollars. This
    # engine is all-in (a BUY commits the whole cash balance), so it tracks the
    # equity curve rather than saying much about the strategy — it is a sanity
    # check on position sizes, NOT a comparison metric between combos. That's why
    # it is shown on the backtest screen but not stored per scan result.
    avg_trade_value: float | None = None

    def __str__(self) -> str:
        pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "n/a"
        return (
            f"Total return:  {self.total_return:.2%}\n"
            f"CAGR:          {self.cagr:.2%}\n"
            f"Max drawdown:  {self.max_drawdown:.2%}\n"
            f"Sharpe ratio:  {self.sharpe_ratio:.2f}\n"
            f"Profit factor: {pf}\n"
            f"Trades:        {self.num_trades} (win rate {self.win_rate:.1%})"
        )


def compute_report(
    equity_curve: pd.Series,
    trades: list,
    periods_per_year: int = TRADING_DAYS_PER_YEAR * MINUTES_PER_TRADING_DAY,
    risk_free_rate: float = 0.0,
) -> PerformanceReport:
    if len(equity_curve) < 2:
        raise ValueError("Equity curve needs at least 2 points to compute metrics")

    returns = equity_curve.pct_change().dropna()

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1

    elapsed_years = (equity_curve.index[-1] - equity_curve.index[0]) / pd.Timedelta(days=365.25)
    ending_ratio = equity_curve.iloc[-1] / equity_curve.iloc[0]
    if elapsed_years > 0 and ending_ratio > 0:
        cagr = ending_ratio ** (1 / elapsed_years) - 1
    else:
        # A busted account (equity at/below zero — possible by at most one
        # commission after fees) has no meaningful compound growth rate.
        cagr = -1.0 if ending_ratio <= 0 else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    max_drawdown = drawdown.min()

    period_rf = risk_free_rate / periods_per_year
    excess_returns = returns - period_rf
    if excess_returns.std() > 0:
        sharpe_ratio = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    else:
        sharpe_ratio = 0.0

    closed_trades = [t for t in trades if t.pnl is not None]
    num_trades = len(closed_trades)
    winners = [t for t in closed_trades if t.pnl > 0]
    losers = [t for t in closed_trades if t.pnl <= 0]
    win_rate = len(winners) / num_trades if num_trades else 0.0

    # Per-trade returns as a fraction of the capital committed to that trade
    # (entry price x shares) — comparable across account sizes, unlike raw $.
    def _trade_return(t) -> float:
        committed = t.entry_price * t.shares
        return t.pnl / committed if committed > 0 else 0.0

    avg_win_pct = sum(_trade_return(t) for t in winners) / len(winners) if winners else None
    avg_loss_pct = sum(_trade_return(t) for t in losers) / len(losers) if losers else None
    expectancy = None
    if num_trades:
        expectancy = win_rate * (avg_win_pct or 0.0) + (1 - win_rate) * (avg_loss_pct or 0.0)

    # Profit factor works in RAW DOLLARS, not per-trade fractions: it asks how many
    # dollars the winners made per dollar the losers gave back, so the totals are
    # the meaningful quantity. Undefined (None) with no trades or no losses.
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = -sum(t.pnl for t in losers)  # losers have pnl <= 0, so this is >= 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    avg_trade_value = (
        sum(t.entry_price * t.shares for t in closed_trades) / num_trades if num_trades else None
    )

    time_in_market = _time_in_market(equity_curve.index, closed_trades)

    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        num_trades=num_trades,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        expectancy=expectancy,
        profit_factor=profit_factor,
        time_in_market=time_in_market,
        avg_trade_value=avg_trade_value,
    )


def _time_in_market(index: pd.DatetimeIndex, closed_trades: list) -> float | None:
    """Fraction of `index`'s bars during which a position was open.

    Counted in BARS via searchsorted rather than by summing wall-clock holding
    time: on minute data an overnight hold spans ~16 hours in which no bar
    exists, and charging the strategy for that exposure would be wrong (and
    could push the fraction above 1). Half-open [entry, exit) so the entry bar
    counts as held and the exit bar does not — an entry and exit on the same
    bar is 0 bars of exposure, which is correct.

    The engine is long-only single-position, so trades never overlap and a
    plain sum needs no interval merging.
    """
    if len(index) == 0 or not closed_trades:
        return None
    entries = [t.entry_time for t in closed_trades if t.exit_time is not None]
    exits = [t.exit_time for t in closed_trades if t.exit_time is not None]
    if not entries:
        return None
    starts = index.searchsorted(pd.DatetimeIndex(entries), side="left")
    ends = index.searchsorted(pd.DatetimeIndex(exits), side="left")
    bars_held = int(np.maximum(ends - starts, 0).sum())
    return bars_held / len(index)


def efficiency_ratio(bars: pd.DataFrame) -> float | None:
    """Kaufman Efficiency Ratio over the SESSION CLOSES of a bar series:
    |net change| / sum(|close-to-close moves|), in [0, 1].

    High = price travelled efficiently in one direction (TRENDING);
    low = lots of movement that went nowhere (RANGE-BOUND / choppy).
    "Trade stocks with a predictable range" — this is the measurable version:
    range strategies (e.g. Bollinger Mean Reversion) want LOW-ER tickers,
    trend strategies want HIGH-ER tickers.

    Computed on session (calendar-day) closes rather than raw bars so the
    value is stable across timespans (a minute-bar scan and a day-bar scan of
    the same window agree). Only comparable across tickers scanned over the
    SAME date window. None if there are fewer than 3 sessions.
    """
    if bars.empty or "close" not in bars:
        return None
    session_closes = bars["close"].groupby(bars.index.date).last()
    if len(session_closes) < 3:
        return None
    net = abs(float(session_closes.iloc[-1]) - float(session_closes.iloc[0]))
    path = float(session_closes.diff().abs().sum())
    if path <= 0:
        return 0.0
    return net / path


# Heuristic ER thresholds for labelling a ticker's behaviour over the scanned
# window. The gap between them is a deliberate "either" gray zone — only
# clearly choppy / clearly trending tickers get a hard label.
RANGE_ER_BELOW = 0.25
TREND_ER_ABOVE = 0.35


def classify_ticker_regime(er: float | None) -> str:
    """"range" / "trend" / "either" from an efficiency ratio (see above)."""
    if er is None:
        return "either"
    if er < RANGE_ER_BELOW:
        return "range"
    if er > TREND_ER_ABOVE:
        return "trend"
    return "either"
