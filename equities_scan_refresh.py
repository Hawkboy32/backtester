"""Refresh the equities scan/roster on current data (2026-08-15).

Last full S&P 500 + Nasdaq-100 scan was runs #39/#40 on 2026-08-03 — 12+ days
stale by the time this runs. Reuses the EXACT same code path as
auto_trader.py's own overnight roster check (_check_roster_overnight, engine.py
around line 274): union of RosterConfig.rescan_universes, RosterConfig.
rescan_strategy_names, RosterConfig.rescan_window_days trailing window, no
strategy_params override (STRATEGY_REGISTRY's defaults already ARE the tuned
values as of 2026-07-31 — VWAP MR entry_deviation_pct=0.3, Bollinger MR
period=15/num_std=3.0). market_calendar="equity" throughout.

SAFE BY CONSTRUCTION: only ever calls compute_recommendation (dry_run=True
under the hood) — writes real scan results to scan_history.db (that's the
intended, designed effect of a scan, same as any manual Scanner-tab run) but
NEVER calls roster.save_pending() or fires a notification. The recommendation
is printed here for a human to read first, same "show findings before
touching anything live" discipline as every other sweep in this project. If
the printed diff should actually go live for dashboard/mobile approval, that
is a deliberate follow-up step, not something this script does on its own.

Run: python equities_scan_refresh.py
"""

from __future__ import annotations

import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import live_trades, roster  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import ENGINE_VERSION  # noqa: E402
from backtester.scan_db import record_scan  # noqa: E402
from backtester.scanner import run_scan  # noqa: E402
from backtester.universe import load_universe  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "equities_scan_refresh_2026-08-15"


def main() -> int:
    state = roster.load_roster()
    cfg = state.config

    tickers = sorted({
        t for universe_name in cfg.rescan_universes
        for t in load_universe(universe_name)["ticker"].tolist()
    })
    # Ends yesterday, not today — same "fully-closed historical days only"
    # discipline as every other sweep here, sidesteps the known Polygon
    # same-day-intraday-data gap regardless of what time this runs.
    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=cfg.rescan_window_days)

    print(f"Universes: {cfg.rescan_universes} -> {len(tickers)} unique tickers")
    print(f"Strategies: {cfg.rescan_strategy_names}")
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()}")
    print(f"engine_version: {ENGINE_VERSION}\n", flush=True)

    t_start = time.monotonic()

    def on_progress(i: int, total: int, ticker: str) -> None:
        if i % 10 == 0 or i == total:
            print(f"[{time.monotonic() - t_start:7.1f}s] {i}/{total}  {ticker}", flush=True)

    client = PolygonClient()
    scan_rows = run_scan(
        tickers=tickers,
        strategy_names=cfg.rescan_strategy_names,
        from_date=from_date.isoformat(),
        to_date=to_date.isoformat(),
        client=client,
        market_calendar="equity",
        checkpoint_path=RESULTS_DIR / "scan_checkpoint.jsonl",
        progress_callback=on_progress,
    )
    print(f"\nScan done in {time.monotonic() - t_start:.1f}s. {len(scan_rows)} results.")

    run_id = record_scan(
        {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "universe": " + ".join(cfg.rescan_universes),
            "num_tickers": len(tickers),
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "multiplier": 1,
            "timespan": "minute",
            "strategy_names": cfg.rescan_strategy_names,
            "engine_version": ENGINE_VERSION,
            "market_calendar": "equity",
            "source": "equities_scan_refresh.py (manual, 2026-08-15)",
        },
        scan_rows,
    )
    print(f"Recorded as scan_history.db run_id={run_id}\n")

    rec = roster.compute_recommendation(state, scan_rows, run_id, live_trades.recent_performance)
    if rec is None:
        print("No roster change recommended — proposed state matches the current live roster exactly.")
        return 0

    print(f"RECOMMENDATION COMPUTED (not yet applied, not yet pushed to dashboard/mobile):")
    print(f"  scan_run_id={rec.scan_run_id}  num_scan_results={rec.num_scan_results}")
    for line in rec.summary:
        print(f"  {line}")
    print(
        "\nThis is a dry-run preview only — roster.save_pending() was NOT called, so nothing "
        "shows up in the dashboard's Auto Trading tab or fires a phone notification yet."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
