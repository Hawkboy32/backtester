"""Same 3-window, 2-year BTC/ETH validation as
crypto_btc_eth_multiwindow_validation.py, rerun at the REAL starting budget
now sitting in CBAPI/KRKAPI (£5 each, used here as $5 - this project does
zero FX conversion anywhere, same simplification as everywhere else GBP
accounts get treated as their raw numeric balance) instead of the earlier
$50 placeholder. Also captures full equity curves (not just end-of-window
summary stats) so the results can be charted, not just tabulated.

Reuses the exact cached 2-year bars from the earlier $50 run - zero new
Polygon calls, this only changes position sizing math and starting_cash.

Run: python crypto_btc_eth_5gbp_multiwindow.py
"""

from __future__ import annotations

import json
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

STARTING_CASH = 5.00  # real balance now in CBAPI/KRKAPI (GBP, used unconverted)
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
OUT_JSON = Path(__file__).resolve().parent / "crypto_btc_eth_5gbp_multiwindow.json"


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
    print(f"Starting cash: ${STARTING_CASH:.2f}")
    print(f"Full range: {start.isoformat()} .. {end.isoformat()} ({TOTAL_DAYS} days)")
    for i, (ws, we) in enumerate(windows):
        print(f"  Window {i+1}: {ws.isoformat()} .. {we.isoformat()} ({(we-ws).days} days)")
    print(flush=True)

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "crypto")

    output: dict = {"starting_cash": STARTING_CASH, "windows": [[w[0].isoformat(), w[1].isoformat()] for w in windows], "series": {}}

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
                    print(f"    {variant_name:8} {params}: end=${end_bal:.4f} Sharpe={report.sharpe_ratio:.3f} trades={report.num_trades}")

                    # Downsample to daily for charting - a minute-level 8-month
                    # curve is way more points than a chart needs, and keeps
                    # the output JSON a sane size.
                    daily = result.equity_curve.resample("1D").last().dropna()
                    series_key = f"{label}/{variant_name}/window{win_idx}"
                    output["series"][series_key] = {
                        "dates": [d.isoformat() for d in daily.index],
                        "values": [round(v, 4) for v in daily.values],
                        "end": round(end_bal, 4),
                        "sharpe": round(report.sharpe_ratio, 4),
                        "trades": report.num_trades,
                    }
                print(flush=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
