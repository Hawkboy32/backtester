"""Validates the user's own observed pattern (2026-08-23): Nasdaq 100 often
spikes hard in one direction right at the open, then reverses within
roughly the first hour. Tests OpeningSpikeFadeStrategy (opening_spike_fade.py)
against I:NDX (Polygon's Nasdaq 100 index ticker - the tradable equivalent
is IG's "US Tech 100" CFD, already linked via MyIGPaper) over just over a
year, across all 3 PositionModes (long-only, short-only, long+short) at
each of the 3 real risk-preset sizing values (Conservative 5%, Moderate
15%, Aggressive 50%) - 9 combinations total, default (untuned) params.

Flat slippage only, not the liquidity-aware model used elsewhere in this
project - an index has no real volume of its own (confirmed 2026-08-23,
Polygon's index bars carry volume=0), so a volume-based slippage model
would be meaningless here; the actual execution vehicle (IG's CFD) has its
own entirely different liquidity/spread characteristics this single-ticker
index backtest can't speak to either way. This is a first-pass "is there
any edge at all" check, same bar every other new strategy in this project
clears before anything gets tuned or funded.

Run: python opening_spike_fade_ndx_test.py
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
from backtester.risk_presets import RISK_PRESETS  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

TICKER = "I:NDX"
STARTING_CASH = 10_000.00
SLIPPAGE_BPS = 2.0
TOTAL_DAYS = 400  # a bit over a year of calendar days, comfortably past 365 trading days


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
    if bars is None or bars.empty:
        print("No bars returned.")
        return 1
    print(f"Got {len(bars)} bars, {bars.index[0]} .. {bars.index[-1]}\n", flush=True)

    ppy = periods_per_year_for_calendar("minute", 1, "equity")  # NYSE/Nasdaq regular session calendar

    strategy = build_strategy("Opening Spike Fade", {})
    print(f"Default params: opening_minutes={strategy.opening_minutes}, "
          f"reversal_window_minutes={strategy.reversal_window_minutes}, "
          f"min_move_pct={strategy.min_move_pct}\n")

    modes = [
        ("Long-only", PositionMode.LONG_ONLY),
        ("Short-only", PositionMode.SHORT_ONLY),
        ("Long+Short", PositionMode.LONG_SHORT),
    ]

    header = f"{'mode':>12}{'preset':>14}{'sizing':>8}{'end $':>14}{'return':>10}{'Sharpe':>9}{'trades':>8}{'win%':>7}"
    print(header)
    print("-" * len(header))
    for mode_label, mode in modes:
        for preset_name, preset in RISK_PRESETS.items():
            sizing_pct = preset["sizing_value"] / 100
            engine = BacktestEngine(
                starting_cash=STARTING_CASH,
                slippage_bps=SLIPPAGE_BPS,
                position_mode=mode,
                dynamic_size_fn=lambda cash, f=sizing_pct: f,
            )
            strategy_instance = build_strategy("Opening Spike Fade", {})
            result = engine.run(bars, strategy_instance)
            try:
                report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
            except ValueError:
                print(f"{mode_label:>12}{preset_name:>14}{preset['sizing_value']:>7.0f}%  not enough trades to score")
                continue
            end_val = result.equity_curve.iloc[-1]
            ret_pct = (end_val / STARTING_CASH - 1) * 100
            wins = sum(1 for t in result.trades if (t.exit_price - t.entry_price) * (1 if t.shares > 0 else -1) > 0)
            win_pct = (wins / len(result.trades) * 100) if result.trades else 0.0
            print(f"{mode_label:>12}{preset_name:>14}{preset['sizing_value']:>7.0f}%"
                  f"{end_val:>14,.2f}{ret_pct:>9.1f}%{report.sharpe_ratio:>9.3f}"
                  f"{report.num_trades:>8}{win_pct:>6.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
