"""Parameter sweep: run the scanner across a grid of parameter values for ONE
strategy on a FIXED window, to compare parameter sets against each other.

The mirror image of walkforward.py: that module holds params fixed and varies
the time window (for performance *consistency*); this module holds the window
fixed and varies params (for performance *comparison*). Neither one on its own
is enough to trust a result — a parameter set that wins here is an in-sample
optimum on ONE window, exactly the shape of result this project has already
learned (twice, the same session this module was built) not to trust at face
value. Always re-run a promising combo through walkforward.py's fold-
consistency check before treating it as a real finding, not just a lucky fit.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

from backtester.data import PolygonClient
from backtester.ranking import aggregate_by_strategy
from backtester.scanner import run_scan
from backtester.universe import load_universe, sample_universe

SweepProgressCallback = Callable[[int, int, dict], None]


def expand_param_grid(param_grid: dict[str, list]) -> list[dict]:
    """Cartesian product of a {param_name: [values, ...]} grid into a list of
    individual {param_name: value} combos, e.g. {"a": [1, 2], "b": [10, 20]}
    -> [{"a":1,"b":10}, {"a":1,"b":20}, {"a":2,"b":10}, {"a":2,"b":20}].
    An empty grid is one combo: the strategy's plain defaults (matching
    run_scan's own "no override" behavior when strategy_params is absent).
    """
    if not param_grid:
        return [{}]
    names = list(param_grid.keys())
    return [dict(zip(names, combo)) for combo in product(*(param_grid[n] for n in names))]


@dataclass
class SweepResult:
    params: dict
    mean_sharpe: float
    mean_return: float
    pct_profitable: float
    num_tickers_tested: int
    num_errors: int
    total_trades: int


def run_param_sweep(
    tickers: list[str],
    strategy_name: str,
    param_grid: dict[str, list],
    from_date: str,
    to_date: str,
    client: PolygonClient,
    progress_callback: SweepProgressCallback | None = None,
    **scan_kwargs,
) -> list[SweepResult]:
    """Run scanner.run_scan once per combination in param_grid (single
    strategy, fixed window, real cost model — the strategy_params seam),
    aggregate each combo's cross-ticker performance the same way
    ranking.aggregate_by_strategy does for a normal scan, sorted best-to-
    worst by mean Sharpe. Extra keyword args (multiplier, timespan,
    slippage_bps, market_calendar, vol_target_enabled, etc.) pass straight
    through to every combo's run_scan call — same pattern as
    walkforward.run_walkforward_scan's **scan_kwargs.
    """
    combos = expand_param_grid(param_grid)
    results: list[SweepResult] = []

    for i, combo in enumerate(combos, start=1):
        if progress_callback:
            progress_callback(i, len(combos), combo)
        rows = run_scan(
            tickers=tickers,
            strategy_names=[strategy_name],
            from_date=from_date,
            to_date=to_date,
            client=client,
            strategy_params={strategy_name: combo},
            **scan_kwargs,
        )
        aggs = aggregate_by_strategy(rows)
        # Exactly one strategy was scanned, so aggregate_by_strategy returns
        # at most one entry back — None only if every ticker errored.
        agg = aggs[0] if aggs else None
        results.append(
            SweepResult(
                params=combo,
                mean_sharpe=agg.mean_sharpe if agg else 0.0,
                mean_return=agg.mean_return if agg else 0.0,
                pct_profitable=agg.pct_profitable if agg else 0.0,
                num_tickers_tested=agg.num_tickers_tested if agg else 0,
                num_errors=agg.num_errors if agg else len(tickers),
                total_trades=agg.total_trades if agg else 0,
            )
        )

    return sorted(results, key=lambda r: r.mean_sharpe, reverse=True)


# ---------------------------------------------------------------------------
# Walk-forward sweep: run_param_sweep above answers "which params win on ONE
# window" — exactly the shape of result this project has repeatedly learned
# not to trust (VWAP MR's default forex result looked fine on one window,
# fell apart on two more; Bollinger MR's num_std=2.0/2.5 forex variants would
# have looked like reasonable guesses without a multi-window check). This
# extends the same idea across MULTIPLE (universe, window) folds in one call,
# so a candidate has to hold up across time and ticker sample, not just win
# once. Built 2026-08-02 after hand-writing this exact loop shape in four
# separate scratchpad scripts the same day.
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardFold:
    """One (universe, window) result for a single sweep candidate."""

    universe: str
    from_date: str
    to_date: str
    sharpe: float | None
    total_trades: int
    scan_seconds: float


@dataclass
class WalkForwardSweepResult:
    strategy_name: str
    label: str
    params: dict
    folds: list[WalkForwardFold] = field(default_factory=list)

    @property
    def fold_sharpes(self) -> list[float]:
        return [f.sharpe for f in self.folds if f.sharpe is not None]

    @property
    def mean_sharpe(self) -> float:
        sharpes = self.fold_sharpes
        return statistics.mean(sharpes) if sharpes else 0.0

    @property
    def std_sharpe(self) -> float:
        sharpes = self.fold_sharpes
        return statistics.pstdev(sharpes) if len(sharpes) > 1 else 0.0

    @property
    def pct_folds_positive(self) -> float:
        sharpes = self.fold_sharpes
        return (sum(1 for s in sharpes if s > 0) / len(sharpes) * 100) if sharpes else 0.0

    @property
    def total_trades(self) -> int:
        return sum(f.total_trades for f in self.folds)


FoldCallback = Callable[[str, str, dict, WalkForwardFold], None]


def run_walkforward_sweep(
    candidates: list[tuple[str, str, dict]],
    windows: list[tuple[str, str]],
    universes: list[str],
    tickers_per_universe: int,
    client: PolygonClient,
    fold_callback: FoldCallback | None = None,
    **scan_kwargs,
) -> list[WalkForwardSweepResult]:
    """Run every (strategy, params) candidate across every (universe, window)
    fold, aggregating into fold-level Sharpe stats (mean/std/% folds
    positive) rather than a single number from one window.

    `candidates`: list of (strategy_name, label, params) — label is a free-
    text description shown in logging/results, e.g. "num_std=4.0, period=20".
    `windows`: list of (from_date, to_date) ISO date strings.
    `universes`: universe names as registered in universe.UNIVERSE_REGISTRY
    (e.g. "S&P 500", "Forex (7 major USD pairs)") — each is sampled down to
    `tickers_per_universe` via universe.sample_universe.
    `fold_callback`, if given, is called after EVERY individual scan, not
    just after each candidate finishes all its folds — a long sweep with
    only per-candidate logging can run for hours with zero visible output
    (learned the hard way 2026-08-02: a 54-scan sweep ran 5h46m before being
    killed, still on its first candidate's log line). Use
    `make_logging_fold_callback()` for a ready-made progress printer, or
    write your own for e.g. writing progress to a file instead of stdout.
    Results are sorted best-to-worst by mean_sharpe, same as run_param_sweep.
    """
    results: list[WalkForwardSweepResult] = []
    for strategy_name, label, params in candidates:
        result = WalkForwardSweepResult(strategy_name=strategy_name, label=label, params=params)
        for universe in universes:
            tickers = sample_universe(load_universe(universe), tickers_per_universe)["ticker"].tolist()
            for from_date, to_date in windows:
                t0 = time.monotonic()
                rows = run_scan(
                    tickers=tickers,
                    strategy_names=[strategy_name],
                    from_date=from_date,
                    to_date=to_date,
                    client=client,
                    strategy_params={strategy_name: params} if params else None,
                    **scan_kwargs,
                )
                scan_secs = time.monotonic() - t0
                aggs = aggregate_by_strategy(rows)
                fold = WalkForwardFold(
                    universe=universe,
                    from_date=from_date,
                    to_date=to_date,
                    sharpe=aggs[0].mean_sharpe if aggs else None,
                    total_trades=sum(r.num_trades or 0 for r in rows),
                    scan_seconds=scan_secs,
                )
                result.folds.append(fold)
                if fold_callback is not None:
                    fold_callback(strategy_name, label, params, fold)
        results.append(result)

    return sorted(results, key=lambda r: r.mean_sharpe, reverse=True)


def make_logging_fold_callback(t_start: float | None = None) -> FoldCallback:
    """Ready-made fold_callback that prints one flushed, elapsed-time-
    prefixed line per scan — the exact pattern hand-written in four separate
    scratchpad scripts on 2026-08-02 before this utility existed. Pass the
    return value as run_walkforward_sweep's fold_callback for a long,
    watchable sweep run from a script (`python -u your_script.py`).
    """
    start = t_start if t_start is not None else time.monotonic()

    def _callback(strategy_name: str, label: str, params: dict, fold: WalkForwardFold) -> None:
        elapsed = time.monotonic() - start
        sharpe_str = round(fold.sharpe, 2) if fold.sharpe is not None else None
        print(
            f"[t+{elapsed:7.1f}s]   [{strategy_name} | {label}] {fold.universe} "
            f"{fold.from_date}..{fold.to_date}: took {fold.scan_seconds:.1f}s, "
            f"trades={fold.total_trades}, sharpe={sharpe_str}",
            flush=True,
        )

    return _callback


def format_sweep_result(result: WalkForwardSweepResult) -> str:
    """One-line summary matching the format hand-written in today's
    scratchpad scripts, for printing after a candidate's sweep completes."""
    return (
        f"[{result.strategy_name}] {result.label}: mean={result.mean_sharpe:.2f} "
        f"std={result.std_sharpe:.2f} pct_positive={result.pct_folds_positive:.0f}% "
        f"(n={len(result.fold_sharpes)}) per_fold={[round(s, 2) for s in result.fold_sharpes]} "
        f"trades={result.total_trades}"
    )
