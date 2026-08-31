"""Multi-window validation for OpeningSpikeFadeStrategy's long-only side -
the tuning pass (opening_spike_fade_tuning.py) found opening_minutes=5
massively beats the default 15, with Sharpe 1.30 (min_move_pct=0.2, 72
trades) and 1.29 (min_move_pct=0.05, 163 trades) on a single ~13-month
window. Same discipline as the BTC/ETH crypto work this session: a
single-window "best params" search risks fitting that window's own noise
(BTC's tuned Bollinger params looked great on one window and lost 88% on
an out-of-sample one) - this checks whether the edge survives split into
independent windows before trusting it.

I:NDX minute data is confirmed available from 2024-06 onward (2023 probed
empty) - splits that ~14+ months into 3 non-overlapping windows, tests
default vs both promising tuned candidates, long-only PositionMode, 100%
sizing throughout (isolates the param question from sizing).

Run: python opening_spike_fade_multiwindow.py
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
    ("default", {"opening_minutes": 15, "reversal_window_minutes": 60, "min_move_pct": 0.1}),
    ("tuned-strict", {"opening_minutes": 5, "reversal_window_minutes": 120, "min_move_pct": 0.2}),
    ("tuned-loose", {"opening_minutes": 5, "reversal_window_minutes": 120, "min_move_pct": 0.05}),
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
    start = date(2024, 6, 1)  # confirmed available (weekday probe returned real rows)
    total_days = (end - start).days
    window_len = total_days // NUM_WINDOWS
    windows = [
        (start + timedelta(days=i * window_len), start + timedelta(days=(i + 1) * window_len - 1))
        for i in range(NUM_WINDOWS)
    ]
    print(f"Full range: {start.isoformat()} .. {end.isoformat()} ({total_days} days)")
    for i, (ws, we) in enumerate(windows, 1):
        print(f"  Window {i}: {ws.isoformat()} .. {we.isoformat()} ({(we - ws).days} days)")
    print(flush=True)

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    scoreboard: dict[str, list[float]] = {name: [] for name, _ in CONFIGS}

    for win_idx, (ws, we) in enumerate(windows, 1):
        print(f"Fetching window {win_idx} ({ws.isoformat()}..{we.isoformat()})...", flush=True)
        try:
            bars = _fetch_with_retry(client, TICKER, ws.isoformat(), we.isoformat())
        except PolygonError as e:
            print(f"  FAILED: {e}\n")
            continue
        if bars is None or bars.empty:
            print("  no bars, skipped\n")
            continue
        print(f"  {len(bars)} bars")

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
                print(f"    {name}: not enough trades to score")
                continue
            end_val = result.equity_curve.iloc[-1]
            print(f"    {name:14} {params}: end=${end_val:,.2f} Sharpe={report.sharpe_ratio:.3f} trades={report.num_trades}")
            scoreboard[name].append(report.sharpe_ratio)
        print(flush=True)

    print("=== SCOREBOARD (Sharpe per window) ===")
    for name, sharpes in scoreboard.items():
        positive = sum(1 for s in sharpes if s > 0)
        print(f"  {name}: {sharpes} -> positive in {positive}/{len(sharpes)} windows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
