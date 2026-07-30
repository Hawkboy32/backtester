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

from dataclasses import dataclass
from itertools import product
from typing import Callable

from backtester.data import PolygonClient
from backtester.ranking import aggregate_by_strategy
from backtester.scanner import run_scan

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
