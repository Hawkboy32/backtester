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

from backtester import events, volatility
from backtester.data import PolygonClient, PolygonError
from backtester.engine import BacktestEngine
from backtester.metrics import (
    MARKET_CALENDARS,
    compute_report,
    efficiency_ratio as compute_efficiency_ratio,
    periods_per_year_for_calendar,
)
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
    avg_conviction: float | None = None  # mean entry-signal conviction across this combo's trades (#25)
    expectancy: float | None = None  # avg per-trade return fraction (see metrics.compute_report)
    profit_factor: float | None = None  # gross profit / gross loss; None when undefined (#36)
    time_in_market: float | None = None  # fraction of bars holding a position (#36)
    efficiency_ratio: float | None = None  # ticker-level Kaufman ER over the window (see metrics.efficiency_ratio)
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
    event_filter_enabled: bool = False,
    market_calendar: str = "equity",
    strategy_params: dict[str, dict] | None = None,
) -> list[ScanResultRow]:
    """strategy_params: optional {strategy_name: {param: value}} overrides,
    merged over that strategy's STRATEGY_REGISTRY defaults (same merge
    build_strategy() already does — {**default_params, **override}). A
    strategy absent from this dict runs with its plain defaults, same as
    before this parameter existed. This is the seam paramsweep.py uses to
    run the same ticker/window/cost setup across a parameter grid without
    duplicating run_scan's fetch/cost/threading machinery.

    vol_target_enabled: when True, fetches a separate daily-bar history per
    ticker (cached independently of the minute/hour/day bars used for the
    actual backtest) and uses a walk-forward GARCH(1,1) volatility regime to
    block new entries during "storm" regimes and scale position size the rest
    of the time (see backtester.volatility). Computed once per ticker, shared
    across all of that ticker's strategy runs. Tickers with insufficient
    daily history (e.g. recent IPOs) fall back to no filter/sizing rather
    than failing the whole ticker.

    event_filter_enabled: when True, suppresses new entries on known
    risk-event days (FOMC — see backtester.events). Off by default so
    existing results stay comparable.

    market_calendar: one of metrics.MARKET_CALENDARS ("equity"/"crypto"/
    "forex") — fixes Sharpe annualization (compute_report) and the GARCH vol
    regime (volatility.compute_regime_table) to the right session-length/
    trading-days-per-year pair. "crypto" is a real 24/7 market (365 days, a
    24h session); "forex" trades a 24h session too but CLOSES on weekends
    (252 days, like equities) — conflating the two overstates forex Sharpe
    by treating Saturday/Sunday as tradable. Leaving this at the default
    "equity" reproduces every existing equities scan's numbers exactly —
    getting this wrong doesn't error, it silently mis-annualizes Sharpe, so
    it must be set explicitly per-scan rather than guessed from the ticker.
    """
    unknown = [name for name in strategy_names if name not in STRATEGY_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}")
    if market_calendar not in MARKET_CALENDARS:
        raise ValueError(f"Unknown market_calendar {market_calendar!r}. Available: {list(MARKET_CALENDARS)}")

    periods_per_year = periods_per_year_for_calendar(timespan, multiplier, market_calendar)
    vol_periods_per_year = MARKET_CALENDARS[market_calendar][1]

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    results: list[ScanResultRow] = []
    tickers_done_strategies: dict[str, set[str]] = {}

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows, existing_map = _load_checkpoint(checkpoint_path)
        results.extend(existing_rows)
        tickers_done_strategies = existing_map

    checkpoint_file = checkpoint_path.open("a", encoding="utf-8") if checkpoint_path else None

    blocked_dates = None
    if event_filter_enabled:
        blocked_dates = events.blocked_dates_in_range(
            pd.Timestamp(from_date).date(), pd.Timestamp(to_date).date()
        )

    def run_one_strategy(
        ticker: str, strategy_name: str, bars, regime_by_date: dict | None, ticker_er: float | None
    ) -> ScanResultRow:
        override = (strategy_params or {}).get(strategy_name)
        params = {**STRATEGY_REGISTRY[strategy_name]["default_params"], **(override or {})}
        try:
            strategy = build_strategy(strategy_name, params=override)
            engine = BacktestEngine(
                starting_cash=starting_cash,
                commission_per_trade=commission_per_trade,
                slippage_bps=slippage_bps,
                regime_by_date=regime_by_date,
                blocked_dates=blocked_dates,
            )
            result = engine.run(bars, strategy)
            report = compute_report(result.equity_curve, result.trades, periods_per_year=periods_per_year)
            convictions = [t.conviction for t in result.trades if t.conviction is not None]
            avg_conviction = sum(convictions) / len(convictions) if convictions else None
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
                avg_conviction=avg_conviction,
                expectancy=report.expectancy,
                profit_factor=report.profit_factor,
                time_in_market=report.time_in_market,
                efficiency_ratio=ticker_er,
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
                    regime_table = volatility.compute_regime_table(
                        daily_bars, target_vol_ann=target_vol_ann, periods_per_year=vol_periods_per_year,
                    )
                    ticker_regime_by_date = volatility.regime_by_date(regime_table)
                except (PolygonError, volatility.InsufficientHistoryError):
                    ticker_regime_by_date = None  # not enough daily history for this ticker; run unfiltered

            # Ticker-level behaviour label, computed once and stamped on every
            # strategy row for this ticker (it's a property of the ticker's
            # price path over the window, not of any strategy).
            ticker_er = compute_efficiency_ratio(bars)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(run_one_strategy, ticker, strategy_name, bars, ticker_regime_by_date, ticker_er)
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
