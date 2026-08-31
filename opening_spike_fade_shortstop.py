"""Tests a much shorter reversal_window (time-stop) directly, following the
trade-analysis finding: winning reversals resolve fast (median 8 min, 90%
within 45 min), while "timed_out" losers (120-min window) average -0.508%
- a tight time-stop should cut losers early without giving up many winners.
Same 3 non-overlapping windows as opening_spike_fade_multiwindow.py, no
min_move_pct filter (the trade analysis found move size barely correlates
with outcome - corr 0.028 - so filtering by it isn't the lever), long-only
PositionMode, opening_minutes=5 (the best single param from tuning).

Run: python opening_spike_fade_shortstop.py
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
from backtester.engine import BacktestEngine, PositionMode  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

TICKER = "I:NDX"
STARTING_CASH = 10_000.00
SLIPPAGE_BPS = 2.0
NUM_WINDOWS = 3

CONFIGS = [
    ("120min (baseline)", {"opening_minutes": 5, "reversal_window_minutes": 120, "min_move_pct": 0.01}),
    ("45min", {"opening_minutes": 5, "reversal_window_minutes": 45, "min_move_pct": 0.01}),
    ("30min", {"opening_minutes": 5, "reversal_window_minutes": 30, "min_move_pct": 0.01}),
    ("20min", {"opening_minutes": 5, "reversal_window_minutes": 20, "min_move_pct": 0.01}),
]


def _fetch_with_retry(client: PolygonClient, ticker: str, from_date: str, to_date: str, attempts: int = 5):
    for i in range(attempts):
        try:
            return client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except PolygonError as e:
            if i == attempts - 1:
                raise
            wait = 65
            print(f"  rate limited, waiting {wait}s (attempt {i + 1}/{attempts}): {e}", flush=True)
            time.sleep(wait)


def main() -> int:
    end = date.today() - timedelta(days=1)
    start = date(2024, 6, 1)
    total_days = (end - start).days
    window_len = total_days // NUM_WINDOWS
    windows = [
        (start + timedelta(days=i * window_len), start + timedelta(days=(i + 1) * window_len - 1))
        for i in range(NUM_WINDOWS)
    ]

    client = PolygonClient()  # cached from earlier runs - should hit disk, not the API
    ppy = periods_per_year_for_calendar("minute", 1, "equity")
    scoreboard: dict[str, list[float]] = {name: [] for name, _ in CONFIGS}

    for win_idx, (ws, we) in enumerate(windows, 1):
        print(f"Window {win_idx}: {ws.isoformat()}..{we.isoformat()}", flush=True)
        try:
            bars = _fetch_with_retry(client, TICKER, ws.isoformat(), we.isoformat())
        except PolygonError as e:
            print(f"  FAILED: {e}\n")
            continue
        if bars is None or bars.empty:
            continue

        for name, params in CONFIGS:
            strategy = build_strategy("Opening Spike Fade", params)
            engine = BacktestEngine(
                starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                position_mode=PositionMode.LONG_ONLY,
                dynamic_size_fn=lambda cash: 1.0,
            )
            result = engine.run(bars, strategy)
            try:
                report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
            except ValueError:
                print(f"    {name}: not enough trades")
                continue
            end_val = result.equity_curve.iloc[-1]
            print(f"    {name:20}: end=${end_val:,.2f} Sharpe={report.sharpe_ratio:.3f} trades={report.num_trades}")
            scoreboard[name].append(report.sharpe_ratio)
        print(flush=True)

    print("=== SCOREBOARD ===")
    for name, sharpes in scoreboard.items():
        positive = sum(1 for s in sharpes if s > 0)
        avg = sum(sharpes) / len(sharpes) if sharpes else 0
        print(f"  {name}: {[round(s, 3) for s in sharpes]} -> positive in {positive}/{len(sharpes)}, avg Sharpe {avg:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
