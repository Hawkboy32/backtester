"""Tunes OpeningSpikeFadeStrategy's long-only side specifically -
opening_spike_fade_ndx_test.py found only the "fade a downward open" side
has any real edge (Sharpe 0.14 default params); the short side is a
consistent loser and isn't worth tuning. Sweeps opening_minutes,
reversal_window_minutes, and min_move_pct against the same ~1yr I:NDX
window, at 100% sizing (isolates the param effect from the sizing
question, matching this project's usual tuning-pass convention), long-only
PositionMode throughout.

Run: python opening_spike_fade_tuning.py
"""

from __future__ import annotations

import sys
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
TOTAL_DAYS = 400

GRID = [
    {"opening_minutes": om, "reversal_window_minutes": rw, "min_move_pct": mp}
    for om in (5, 10, 15, 30)
    for rw in (30, 60, 90, 120)
    for mp in (0.05, 0.1, 0.2, 0.3)
]


def main() -> int:
    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=TOTAL_DAYS)
    print(f"Fetching {TICKER}, {from_date.isoformat()} .. {to_date.isoformat()}...", flush=True)

    client = PolygonClient()
    try:
        bars = client.get_aggregates(TICKER, from_date.isoformat(), to_date.isoformat(), 1, "minute")
    except PolygonError as e:
        print(f"FAILED: {e}")
        return 1
    print(f"Got {len(bars)} bars\n", flush=True)

    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    default_params = {"opening_minutes": 15, "reversal_window_minutes": 60, "min_move_pct": 0.1}
    rows = []
    for params in GRID:
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
            continue
        end_val = result.equity_curve.iloc[-1]
        rows.append({
            "params": params, "end": end_val, "sharpe": report.sharpe_ratio,
            "trades": report.num_trades,
        })
        print(f"  {params}: end=${end_val:,.2f} Sharpe={report.sharpe_ratio:.3f} trades={report.num_trades}", flush=True)

    rows.sort(key=lambda r: r["sharpe"], reverse=True)
    default_row = next((r for r in rows if r["params"] == default_params), None)
    print(f"\nDEFAULT {default_params}: Sharpe={default_row['sharpe']:.3f}" if default_row else "\ndefault not in grid")
    print("\nTop 10 by Sharpe:")
    for r in rows[:10]:
        print(f"  {r['params']}: end=${r['end']:,.2f} Sharpe={r['sharpe']:.3f} trades={r['trades']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
