"""Walk-forward validation: run the scanner across multiple sequential time
folds and check whether a strategy's performance holds up consistently
across periods, rather than being a fluke of one window.

This does not optimize parameters per fold — the strategies here use fixed
default params, so this isn't a parameter search. It's about performance
*consistency* across time: a strategy that looks great in fold 1 and
terrible in fold 3 is a red flag the single-window scanner can't surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from backtester.data import PolygonClient
from backtester.ranking import aggregate_by_strategy
from backtester.scanner import ScanResultRow, run_scan

FoldProgressCallback = Callable[[int, int, str, str], None]


def split_date_range(from_date: str, to_date: str, num_folds: int) -> list[tuple[str, str]]:
    if num_folds < 1:
        raise ValueError("num_folds must be >= 1")

    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    total_days = (end - start).days
    if total_days < num_folds:
        raise ValueError(f"Date range ({total_days} days) too short to split into {num_folds} folds")

    fold_days = total_days // num_folds
    folds: list[tuple[str, str]] = []
    cursor = start
    for i in range(num_folds):
        fold_start = cursor
        fold_end = cursor + timedelta(days=fold_days) if i < num_folds - 1 else end
        folds.append((fold_start.strftime("%Y-%m-%d"), fold_end.strftime("%Y-%m-%d")))
        cursor = fold_end
    return folds


def run_walkforward_scan(
    tickers: list[str],
    strategy_names: list[str],
    from_date: str,
    to_date: str,
    num_folds: int,
    client: PolygonClient,
    checkpoint_dir: Path | str | None = None,
    fold_progress_callback: FoldProgressCallback | None = None,
    **scan_kwargs,
) -> dict[str, list[ScanResultRow]]:
    """Run backtester.scanner.run_scan once per fold. Extra keyword args
    (multiplier, timespan, starting_cash, progress_callback, result_callback,
    etc.) pass straight through to each fold's run_scan call.
    """
    folds = split_date_range(from_date, to_date, num_folds)
    checkpoint_dir_path = Path(checkpoint_dir) if checkpoint_dir else None

    fold_results: dict[str, list[ScanResultRow]] = {}
    for i, (fold_from, fold_to) in enumerate(folds, start=1):
        fold_label = f"fold_{i}_{fold_from}_to_{fold_to}"
        if fold_progress_callback:
            fold_progress_callback(i, len(folds), fold_from, fold_to)

        checkpoint_path = checkpoint_dir_path / f"{fold_label}.jsonl" if checkpoint_dir_path else None
        results = run_scan(
            tickers=tickers,
            strategy_names=strategy_names,
            from_date=fold_from,
            to_date=fold_to,
            client=client,
            checkpoint_path=checkpoint_path,
            **scan_kwargs,
        )
        fold_results[fold_label] = results

    return fold_results


@dataclass
class WalkforwardConsistency:
    strategy_name: str
    num_folds: int
    mean_sharpe_across_folds: float
    std_sharpe_across_folds: float
    pct_folds_profitable: float
    worst_fold_sharpe: float
    best_fold_sharpe: float


def aggregate_walkforward(fold_results: dict[str, list[ScanResultRow]]) -> list[WalkforwardConsistency]:
    """Per strategy, aggregate mean Sharpe/return within each fold, then look
    at how that per-fold number varies across folds. A strategy with high
    mean Sharpe but also high std across folds is inconsistent — that's
    exactly what a single-window scan can't tell you.
    """
    per_strategy_fold_sharpe: dict[str, list[float]] = {}
    per_strategy_fold_return: dict[str, list[float]] = {}

    for rows in fold_results.values():
        for agg in aggregate_by_strategy(rows):
            per_strategy_fold_sharpe.setdefault(agg.strategy_name, []).append(agg.mean_sharpe)
            per_strategy_fold_return.setdefault(agg.strategy_name, []).append(agg.mean_return)

    consistency: list[WalkforwardConsistency] = []
    for strategy_name, sharpes in per_strategy_fold_sharpe.items():
        returns = per_strategy_fold_return[strategy_name]
        consistency.append(
            WalkforwardConsistency(
                strategy_name=strategy_name,
                num_folds=len(sharpes),
                mean_sharpe_across_folds=float(np.mean(sharpes)),
                std_sharpe_across_folds=float(np.std(sharpes)),
                pct_folds_profitable=float(np.mean([r > 0 for r in returns])),
                worst_fold_sharpe=float(np.min(sharpes)),
                best_fold_sharpe=float(np.max(sharpes)),
            )
        )

    return sorted(consistency, key=lambda c: c.mean_sharpe_across_folds, reverse=True)
