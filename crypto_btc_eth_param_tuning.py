"""BTC/ETH-specific param tuning - task #6 from the crypto funding-plan work.

Forex needed its own tuned Bollinger params (num_std=4.0, period=20) rather
than the equities-derived default (num_std=3.0, period=15) - see
CLAUDE_NOTES.txt's Phase 2 entries. crypto_liquidity_stress_test.py (this
session) narrowed the 15-pair crypto universe down to BTC and ETH only -
every other pair showed severe-to-catastrophic liquidity drag at realistic
sizing. This script checks whether BTC/ETH specifically benefit from their
own tuned variant the same way forex did, rather than assuming the equities
default transfers.

Reuses the exact same 90-day cached bars + liquidity slippage model as the
stress test (same tickers, same window - zero new Polygon calls). Ranks by
liquidity-adjusted Sharpe at 100% sizing, since that's how the live system
actually runs (100% global sizing_value, per-ticker caps doing the real
limiting - see project_trading_bot_liquidity_and_tax memory) rather than an
arbitrary mid-grid point.

Run: python crypto_btc_eth_param_tuning.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

STARTING_CASH = 50.00
BASE_BPS, IMPACT_BPS_PER_PCT, CAP_BPS = 2.0, 8.0, 500.0
TICKERS = ["X:BTCUSD", "X:ETHUSD"]

BOLLINGER_GRID = [
    {"period": p, "num_std": s}
    for p in (10, 15, 20, 30)
    for s in (2.0, 2.5, 3.0, 3.5, 4.0)
]
VWAP_GRID = [
    {"min_bars": m, "entry_deviation_pct": d}
    for m in (5, 10, 15)
    for d in (0.15, 0.2, 0.3, 0.5, 0.75)
]


def liquidity_slippage(shares: float, bar_volume: float) -> float:
    if bar_volume <= 0:
        return CAP_BPS
    return min(BASE_BPS + (100.0 * shares / bar_volume) * IMPACT_BPS_PER_PCT, CAP_BPS)


def main() -> int:
    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=90)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (should hit the on-disk cache, zero new fetches)\n")

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "crypto")

    for ticker in TICKERS:
        try:
            bars = client.get_aggregates(ticker, from_date.isoformat(), to_date.isoformat(), 1, "minute")
        except PolygonError as e:
            print(f"{ticker}: FAILED to fetch - {e}")
            continue
        if bars is None or bars.empty:
            print(f"{ticker}: no bars")
            continue

        for strategy_name, grid in (("Bollinger Mean Reversion", BOLLINGER_GRID), ("VWAP Mean Reversion", VWAP_GRID)):
            print(f"=== {ticker}/{strategy_name} (default params, then grid, all at 100% sizing) ===")
            rows = []
            for params in grid:
                strat = build_strategy(strategy_name, params)
                engine = BacktestEngine(
                    starting_cash=STARTING_CASH, slippage_bps=BASE_BPS,
                    dynamic_size_fn=lambda cash: 1.0,
                    variable_slippage_fn=liquidity_slippage,
                )
                result = engine.run(bars, strat)
                try:
                    report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
                except ValueError:
                    continue
                end = result.equity_curve.iloc[-1]
                rows.append({"params": params, "end": end, "sharpe": report.sharpe_ratio, "trades": report.num_trades})

            rows.sort(key=lambda r: r["sharpe"], reverse=True)
            default_params = {"period": 15, "num_std": 3.0} if strategy_name == "Bollinger Mean Reversion" else {"min_bars": 5, "entry_deviation_pct": 0.3}
            default_row = next((r for r in rows if r["params"] == default_params), None)
            print(f"  DEFAULT {default_params}: end=${default_row['end']:.2f} Sharpe={default_row['sharpe']:.3f} trades={default_row['trades']}" if default_row else "  default not in grid")
            print("  Top 5 by liquidity-adjusted Sharpe:")
            for r in rows[:5]:
                print(f"    {r['params']}: end=${r['end']:.2f} Sharpe={r['sharpe']:.3f} trades={r['trades']}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
