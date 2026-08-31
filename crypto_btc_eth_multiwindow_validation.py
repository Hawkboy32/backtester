"""BTC/ETH multi-window validation - the final gate before deciding whether
to fund CBAPI/KRKAPI. crypto_btc_eth_param_tuning.py found a tuned param set
(Bollinger: period=30/num_std=2.0, VWAP: entry_deviation_pct=0.15) that beat
the equities-derived defaults by a wide margin - but on a SINGLE 90-day
window, which risks fitting to that window's own noise rather than a real,
durable edge (the same overfitting risk this project guarded against
everywhere else - equities/forex/crypto Phase 1 all got multi-window
validation before being trusted).

METHOD
Full ~2-year history split into 3 non-overlapping windows (oldest to
newest), same "does the edge survive out of a single sample" bar as
Phase 1's original crypto validation. For each window, for BTC and ETH, for
both Bollinger Mean Reversion and VWAP Mean Reversion, runs BOTH the
default and the tuned params side by side at 100% sizing (matching how the
live system actually runs) under the same liquidity-aware slippage model
used throughout this exercise. Reports whether tuned beats default in each
window individually, not just on average - a real edge should win most or
all windows, not just look good in aggregate.

Run: python crypto_btc_eth_multiwindow_validation.py
"""

from __future__ import annotations

import sys
import time
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

CONFIGS = {
    "Bollinger Mean Reversion": [
        ("default", {"period": 15, "num_std": 3.0}),
        ("tuned", {"period": 30, "num_std": 2.0}),
    ],
    "VWAP Mean Reversion": [
        ("default", {"min_bars": 5, "entry_deviation_pct": 0.3}),
        ("tuned", {"min_bars": 5, "entry_deviation_pct": 0.15}),
    ],
}

NUM_WINDOWS = 3
TOTAL_DAYS = 730


def liquidity_slippage(shares: float, bar_volume: float) -> float:
    if bar_volume <= 0:
        return CAP_BPS
    return min(BASE_BPS + (100.0 * shares / bar_volume) * IMPACT_BPS_PER_PCT, CAP_BPS)


def _fetch_with_retry(client: PolygonClient, ticker: str, from_date: str, to_date: str, attempts: int = 5):
    for i in range(attempts):
        try:
            return client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except PolygonError as e:
            if i == attempts - 1:
                raise
            wait = 65
            print(f"    rate limited fetching {ticker}, waiting {wait}s (attempt {i + 1}/{attempts}): {e}", flush=True)
            time.sleep(wait)


def main() -> int:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=TOTAL_DAYS)
    window_len = TOTAL_DAYS // NUM_WINDOWS
    windows = []
    for i in range(NUM_WINDOWS):
        w_start = start + timedelta(days=i * window_len)
        w_end = w_start + timedelta(days=window_len - 1)
        windows.append((w_start, w_end))
    print(f"Full range: {start.isoformat()} .. {end.isoformat()} ({TOTAL_DAYS} days)")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i+1}: {ws.isoformat()} .. {we.isoformat()} ({(we-ws).days} days)")
    print(flush=True)

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "crypto")

    scoreboard: dict[str, list[int]] = {}  # "ticker/strategy" -> [wins for tuned across windows]

    for ticker in TICKERS:
        for win_idx, (w_start, w_end) in enumerate(windows, start=1):
            print(f"Fetching {ticker} window {win_idx} ({w_start.isoformat()}..{w_end.isoformat()})...", flush=True)
            try:
                bars = _fetch_with_retry(client, ticker, w_start.isoformat(), w_end.isoformat())
            except PolygonError as e:
                print(f"  FAILED: {e}\n")
                continue
            if bars is None or bars.empty or len(bars) < 100:
                print("  too few bars, skipped\n")
                continue

            for strategy_name, variants in CONFIGS.items():
                label = f"{ticker}/{strategy_name}"
                print(f"  === {label} (window {win_idx}) ===")
                window_results = {}
                for variant_name, params in variants:
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
                        print(f"    {variant_name} {params}: not enough trades to score")
                        continue
                    end_bal = result.equity_curve.iloc[-1]
                    window_results[variant_name] = report.sharpe_ratio
                    print(f"    {variant_name:8} {params}: end=${end_bal:.2f} Sharpe={report.sharpe_ratio:.3f} trades={report.num_trades}")

                if "default" in window_results and "tuned" in window_results:
                    tuned_wins = window_results["tuned"] > window_results["default"]
                    scoreboard.setdefault(label, []).append(1 if tuned_wins else 0)
                    print(f"    -> tuned {'BEATS' if tuned_wins else 'loses to'} default this window")
                print(flush=True)

    print("\n=== SCOREBOARD (tuned vs default, wins per window) ===")
    for label, wins in scoreboard.items():
        print(f"  {label}: tuned won {sum(wins)}/{len(wins)} windows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
