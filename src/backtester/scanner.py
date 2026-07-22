"""Concurrent multi-ticker, multi-strategy backtest scanner.

Data fetching is sequential per ticker (the Polygon client's own rate
limiter throttles it); once a ticker's bars are in memory, every requested
strategy is backtested against them concurrently via a thread pool. Results
are checkpointed to a JSONL file as they complete, so an interrupted scan
can be resumed without re-fetching or re-running tickers already done.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from backtester import volatility
from backtester.data import PolygonClient, PolygonError
from backtester.engine import BacktestEngine
from backtester.metrics import compute_report
from backtester.strategies import STRATEGY_REGISTRY, build_strategy


@dataclass
class ScanResultRow:
    ticker: str
    strategy_name: str
    params: dict
    total_return: float | None = None
    cagr: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    num_trades: int | None = None
    win_rate: float | None = None
    num_bars: int | None = None
    error: str | None = None


ProgressCallback = Callable[[int, int, str], None]


def _load_checkpoint(checkpoint_path: Path) -> tuple[list[ScanResultRow], dict[str, set[str]]]:
    rows: list[ScanResultRow] = []
    if not checkpoint_path.exists():
        return rows, {}

    with checkpoint_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rows.append(ScanResultRow(**data))

    tickers_seen: dict[str, set[str]] = {}
    for row in rows:
        tickers_seen.setdefault(row.ticker, set()).add(row.strategy_name)
    return rows, tickers_seen


def run_scan(
    tickers: list[str],
    strategy_names: list[str],
    from_date: str,
    to_date: str,
    client: PolygonClient,
    multiplier: int = 1,
    timespan: str = "minute",
    starting_cash: float = 100_000.0,
    # Alpaca-like defaults: US stocks/ETFs are commission-free there; the real
    # cost is crossing the spread (~1-2 bps on liquid names) plus sub-bp
    # sell-side regulatory fees — modeled together as 2 bps slippage per side.
    commission_per_trade: float = 0.0,
    slippage_bps: float = 2.0,
    max_workers: int = 4,
    checkpoint_path: Path | str | None = None,
    progress_callback: ProgressCallback | None = None,
    result_callback: Callable[[ScanResultRow], None] | None = None,
    vol_target_enabled: bool = False,
    target_vol_ann: float = volatility.DEFAULT_TARGET_VOL_ANN,
    daily_lookback_days: int = 1100,
) -> list[ScanResultRow]:
    """vol_target_enabled: when True, fetches a separate daily-bar history per
    ticker (cached independently of the minute/hour/day bars used for the
    actual backtest) and uses a walk-forward GARCH(1,1) volatility regime to
    block new entries during "storm" regimes and scale position size the rest
    of the time (see backtester.volatility). Computed once per ticker, shared
    across all of that ticker's strategy runs. Tickers with insufficient
    daily history (e.g. recent IPOs) fall back to no filter/sizing rather
    than failing the whole ticker.
    """
    unknown = [name for name in strategy_names if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    results: list[ScanResultRow] = []
    tickers_done_strategies: dict[str, set[str]] = {}

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows, existing_map = _load_checkpoint(checkpoint_path)
        results.extend(existing_rows)
        tickers_done_strategies = existing_map

    checkpoint_file = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None

    def run_one_strategy(ticker: str, strategy_name: str, bars, regime_by_date: dict | None) -> ScanResultRow:
        params = STRATEGY_REGISTRY[strategy_name]["default_params"]
        try:
            strategy = build_strategy(strategy_name)
            engine = BacktestEngine(
                starting_cash=starting_cash,
                commission_per_trade=commission_per_trade,
                slippage_bps=slippage_bps,
                regime_by_date=regime_by_date,
            )
            result = engine.run(bars, strategy)
            report = compute_report(result.equity_curve, result.trades)
            return ScanResultRow(
                ticker=ticker,
                strategy_name=strategy_name,
                params=params,
                total_return=report.total_return,
                cagr=report.cagr,
                max_drawdown=report.max_drawdown,
                sharpe_ratio=report.sharpe_ratio,
                num_trades=report.num_trades,
                win_rate=report.win_rate,
                num_bars=len(bars),
            )
        except Exception as e:  # noqa: BLE001
            return ScanResultRow(ticker=ticker, strategy_name=strategy_name, params=params, error=str(e))

    try:
        for i, ticker in enumerate(tickers, start=1):
            already_done = tickers_done_strategies.get(ticker, set())
            pending_strategies = [s for s in strategy_names if s not in already_done]

            if progress_callback:
                progress_callback(i, len(tickers), ticker)

            if not pending_strategies:
                continue

            try:
                bars = client.get_aggregates(
                    ticker=ticker,
                    from_date=from_date,
                    to_date=to_date,
                    multiplier=multiplier,
                    timespan=timespan,
                )
            except PolygonError as e:
                for strategy_name in pending_strategies:
                    row = ScanResultRow(
                        ticker=ticker,
                        strategy_name=strategy_name,
                        params=STRATEGY_REGISTRY[strategy_name]["default_params"],
                        error=f"data fetch failed: {e}",
                    )
                    results.append(row)
                    if result_callback:
                        result_callback(row)
                    if checkpoint_file:
                        checkpoint_file.write(json.dumps(asdict(row)) + "\n")
                        checkpoint_file.flush()
                continue

            if bars.empty:
                for strategy_name in pending_strategies:
                    row = ScanResultRow(
                        ticker=ticker,
                        strategy_name=strategy_name,
                        params=STRATEGY_REGISTRY[strategy_name]["default_params"],
                        error="no bars returned",
                    )
                    results.append(row)
                    if result_callback:
                        result_callback(row)
                    if checkpoint_file:
                        checkpoint_file.write(json.dumps(asdict(row)) + "\n")
                        checkpoint_file.flush()
                continue

            ticker_regime_by_date = None
            if vol_target_enabled:
                try:
                    daily_from = (pd.Timestamp(from_date) - pd.Timedelta(days=daily_lookback_days)).date().isoformat()
                    daily_bars = client.get_aggregates(
                        ticker=ticker, from_date=daily_from, to_date=to_date, multiplier=1, timespan="day",
                    )
                    regime_table = volatility.compute_regime_table(daily_bars, target_vol_ann=target_vol_ann)
                    ticker_regime_by_date = volatility.regime_by_date(regime_table)
                except (PolygonError, volatility.InsufficientHistoryError):
                    ticker_regime_by_date = None  # not enough daily history for this ticker; run unfiltered

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(run_one_strategy, ticker, strategy_name, bars, ticker_regime_by_date)
                    for strategy_name in pending_strategies
                ]
                for future in futures:
                    row = future.result()
                    results.append(row)
                    if result_callback:
                        result_callback(row)
                    if checkpoint_file:
                        checkpoint_file.write(json.dumps(asdict(row)) + "\n")
                        checkpoint_file.flush()
    finally:
        if checkpoint_file:
            checkpoint_file.close()

    return results
