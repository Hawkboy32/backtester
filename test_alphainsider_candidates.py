"""Rigorous test of the 3 AlphaInsider-sourced candidates added 2026-09-06
(DMI/ADX Trend, Linear Regression Channel, DPO Mean-Reversion) against the
SAME universe, window, and methodology the roster's own real rescan uses -
S&P 500 + Nasdaq-100, roster.RosterConfig's own rescan_window_days (30),
minute bars, market_calendar="equity" - not a smaller ad hoc sample, so the
result is directly comparable to how every currently-live strategy was
actually vetted.

Two already-proven, currently-live strategies (VWAP Mean Reversion, Bollinger
Mean Reversion) run alongside the 3 candidates as a baseline - raw numbers on
their own don't say whether a candidate is actually competitive with what's
already trusted.

Writes real results to scan_history.db via record_scan (same as any other
real scan) so this becomes reusable history, not a throwaway. Checkpointed
(resumable) given the ~600-ticker universe. Never touches control.json,
roster.json, or anything live - read-only against Polygon, write-only to
scan_history.db.

Run: python test_alphainsider_candidates.py
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import roster  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import ENGINE_VERSION  # noqa: E402
from backtester.scan_db import record_scan  # noqa: E402
from backtester.scanner import run_scan  # noqa: E402
from backtester.universe import load_universe  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parent / "results" / "alphainsider_candidates_checkpoint.jsonl"
CANDIDATES = ["DMI/ADX Trend", "Linear Regression Channel", "DPO Mean-Reversion"]
BASELINE = ["VWAP Mean Reversion", "Bollinger Mean Reversion"]


def main() -> int:
    cfg = roster.load_roster().config  # reuse the REAL, currently-live roster config
    tickers = sorted({
        t for universe_name in cfg.rescan_universes
        for t in load_universe(universe_name)["ticker"].tolist()
    })
    to_date = date.today()
    from_date = to_date - timedelta(days=cfg.rescan_window_days)
    strategy_names = CANDIDATES + BASELINE

    print(f"Universe: {len(tickers)} tickers ({', '.join(cfg.rescan_universes)})")
    print(f"Window:   {from_date} -> {to_date} ({cfg.rescan_window_days} days, matches the roster's own rescan)")
    print(f"Strategies: {strategy_names}")
    print()

    t0 = time.monotonic()

    def progress(i: int, total: int, ticker: str) -> None:
        if i % 25 == 0 or i == total - 1:
            elapsed = time.monotonic() - t0
            print(f"  [{i + 1}/{total}] {ticker}  ({elapsed:.0f}s elapsed)", flush=True)

    rows = run_scan(
        tickers=tickers,
        strategy_names=strategy_names,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        client=PolygonClient(),
        market_calendar="equity",
        checkpoint_path=CHECKPOINT,
        progress_callback=progress,
    )

    run_id = record_scan(
        {
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
            "universe": " + ".join(cfg.rescan_universes),
            "num_tickers": len(tickers),
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "multiplier": 1,
            "timespan": "minute",
            "strategy_names": strategy_names,
            "engine_version": ENGINE_VERSION,
        },
        rows,
    )
    print(f"\nScan recorded: run_id={run_id}, {len(rows)} rows")

    print("\n=== Per-strategy aggregate (across every ticker that produced at least 1 trade) ===")
    print(f"{'Strategy':<28} {'Tickers':>8} {'Trades':>7} {'AvgRet%':>9} {'AvgSharpe':>10} {'AvgMaxDD%':>10} {'AvgWinRate%':>12} {'AvgPF':>7}")
    for name in strategy_names:
        matching = [r for r in rows if r.strategy_name == name and r.error is None and (r.num_trades or 0) > 0]
        if not matching:
            print(f"{name:<28} {'(no trades fired across the whole universe/window)':>60}")
            continue
        n = len(matching)
        avg = lambda field: sum((getattr(r, field) or 0.0) for r in matching) / n  # noqa: E731
        total_trades = sum((r.num_trades or 0) for r in matching)
        print(
            f"{name:<28} {n:>8} {total_trades:>7} {avg('total_return') * 100:>9.2f} "
            f"{avg('sharpe_ratio'):>10.2f} {avg('max_drawdown') * 100:>10.2f} "
            f"{avg('win_rate') * 100:>12.1f} {avg('profit_factor'):>7.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
