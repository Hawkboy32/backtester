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

    def __str__(self) -> str:
        return (
            f"Total return:  {self.total_return:.2%}\n"
            f"CAGR:          {self.cagr:.2%}\n"
            f"Max drawdown:  {self.max_drawdown:.2%}\n"
            f"Sharpe ratio:  {self.sharpe_ratio:.2f}\n"
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
    win_rate = len(winners) / num_trades if num_trades else 0.0

    return PerformanceReport(
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        num_trades=num_trades,
        win_rate=win_rate,
    )
