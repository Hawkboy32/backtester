"""Run the SMA crossover strategy backtest against real Polygon.io minute bars.

Usage:
    python examples/run_sma_backtest.py AAPL 2025-01-01 2025-06-01
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from backtester.data import PolygonClient
from backtester.engine import BacktestEngine
from backtester.metrics import compute_report
from backtester.strategies.sma_crossover import SmaCrossoverStrategy


def main() -> None:
    load_dotenv()

    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} TICKER FROM_DATE TO_DATE")
        sys.exit(1)

    ticker, from_date, to_date = sys.argv[1], sys.argv[2], sys.argv[3]

    client = PolygonClient()
    bars = client.get_aggregates(
        ticker=ticker,
        from_date=from_date,
        to_date=to_date,
        multiplier=1,
        timespan="minute",
    )
    print(f"Fetched {len(bars)} minute bars for {ticker} from {from_date} to {to_date}")

    strategy = SmaCrossoverStrategy(fast_window=20, slow_window=50)
    engine = BacktestEngine(starting_cash=100_000.0, commission_per_trade=1.0, slippage_bps=1.0)
    result = engine.run(bars, strategy)

    report = compute_report(result.equity_curve, result.trades)
    print(report)


if __name__ == "__main__":
    main()
