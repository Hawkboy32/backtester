"""Metric-based ranking of scan results.

Not a trained model — a transparent, configurable weighted score over the
backtest metrics themselves (Sharpe, return, drawdown, win rate). Each
metric is min-max normalized across the valid (non-error) results before
weighting, so strategies/tickers are compared on a common 0-1 scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtester.scanner import ScanResultRow

DEFAULT_WEIGHTS = {
    "sharpe_ratio": 0.4,
    "total_return": 0.3,
    "max_drawdown": 0.2,
    "win_rate": 0.1,
}


@dataclass
class StrategyAggregate:
    strategy_name: str
    num_tickers_tested: int
    num_errors: int
    mean_sharpe: float
    median_sharpe: float
    mean_return: float
    pct_profitable: float
    mean_max_drawdown: float
    total_trades: int


def _results_to_frame(rows: list[ScanResultRow]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": r.ticker,
                "strategy_name": r.strategy_name,
                "total_return": r.total_return,
                "cagr": r.cagr,
                "max_drawdown": r.max_drawdown,
                "sharpe_ratio": r.sharpe_ratio,
                "num_trades": r.num_trades,
                "win_rate": r.win_rate,
                "num_bars": r.num_bars,
                # Display/learning columns — NOT part of the score weights.
                "expectancy": r.expectancy,
                "efficiency_ratio": r.efficiency_ratio,
                "error": r.error,
            }
            for r in rows
        ]
    )


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi <= lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def rank_combos(rows: list[ScanResultRow], weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Return a DataFrame of valid (non-error) results with a composite `score`,
    sorted descending. Rows that errored (no data, fetch failure, etc.) are excluded.
    """
    weights = weights or DEFAULT_WEIGHTS
    df = _results_to_frame(rows)
    valid = df[df["error"].isna()].copy()
    if valid.empty:
        return valid.assign(score=[])

    valid["score"] = (
        weights.get("sharpe_ratio", 0) * _minmax(valid["sharpe_ratio"])
        + weights.get("total_return", 0) * _minmax(valid["total_return"])
        + weights.get("max_drawdown", 0) * _minmax(valid["max_drawdown"])
        + weights.get("win_rate", 0) * valid["win_rate"].fillna(0)
    )
    return valid.sort_values("score", ascending=False).reset_index(drop=True)


def aggregate_by_strategy(rows: list[ScanResultRow]) -> list[StrategyAggregate]:
    """Per-strategy performance across every ticker it was tested on — this is
    the more useful view for 'which strategies should we implement', since a
    single top ticker×strategy combo can just be noise.
    """
    df = _results_to_frame(rows)
    aggregates: list[StrategyAggregate] = []

    for strategy_name, group in df.groupby("strategy_name"):
        valid = group[group["error"].isna()]
        errors = group[group["error"].notna()]
        if valid.empty:
            aggregates.append(
                StrategyAggregate(
                    strategy_name=strategy_name,
                    num_tickers_tested=0,
                    num_errors=len(errors),
                    mean_sharpe=0.0,
                    median_sharpe=0.0,
                    mean_return=0.0,
                    pct_profitable=0.0,
                    mean_max_drawdown=0.0,
                    total_trades=0,
                )
            )
            continue

        aggregates.append(
            StrategyAggregate(
                strategy_name=strategy_name,
                num_tickers_tested=len(valid),
                num_errors=len(errors),
                mean_sharpe=float(valid["sharpe_ratio"].mean()),
                median_sharpe=float(valid["sharpe_ratio"].median()),
                mean_return=float(valid["total_return"].mean()),
                pct_profitable=float((valid["total_return"] > 0).mean()),
                mean_max_drawdown=float(valid["max_drawdown"].mean()),
                total_trades=int(valid["num_trades"].sum()),
            )
        )

    return sorted(aggregates, key=lambda a: a.mean_sharpe, reverse=True)
