"""One-off overnight sweep for the risk-dial's vol-target-sizing axis (2b) —
see CLAUDE_NOTES.txt's "Conservative/aggressive risk dial" entry for the full
design discussion. Holds each asset class's already walk-forward-tuned entry
params FIXED (the axis-2a work Phase 1-4 already did) and varies target_vol_ann
only, across the same 3-window walk-forward discipline used throughout this
project. Results are appended to risk_dial_sweep_results.jsonl as each run
completes, not held in memory until the end — a past sweep here ran 5h46m
with zero visible output because only per-candidate (not per-scan) progress
was logged; this logs every single scan as it finishes instead.

Run: python risk_dial_vol_sweep.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.data import PolygonClient  # noqa: E402
from backtester.ranking import aggregate_by_strategy  # noqa: E402
from backtester.scanner import run_scan  # noqa: E402

WINDOWS = [
    ("2026-04-16", "2026-05-19"),
    ("2026-05-20", "2026-06-22"),
    ("2026-06-23", "2026-07-26"),
]

# Centered on volatility.DEFAULT_TARGET_VOL_ANN (20%) — half and 1.5x either side.
VOL_TARGET_GRID = [10.0, 15.0, 20.0, 25.0, 30.0]

BUCKETS = [
    {
        "label": "Equities (TSLA/AAPL)",
        "tickers": ["TSLA", "AAPL"],
        "market_calendar": "equity",
        # Phase 4-tuned equity params (CLAUDE_NOTES.txt 2026-08-03).
        "strategies": {
            "VWAP Mean Reversion": {"entry_deviation_pct": 0.3},
            "Bollinger Mean Reversion": {"period": 15, "num_std": 3.0},
        },
    },
    {
        "label": "Gold/Silver (GLD/SLV)",
        "tickers": ["GLD", "SLV"],
        "market_calendar": "equity",
        # ETF proxies trade like equities — same tuned params as the equity bucket.
        "strategies": {
            "VWAP Mean Reversion": {"entry_deviation_pct": 0.3},
            "Bollinger Mean Reversion": {"period": 15, "num_std": 3.0},
        },
    },
    {
        "label": "Forex (GBPUSD/EURGBP)",
        "tickers": ["C:GBPUSD", "C:EURGBP"],
        "market_calendar": "forex",
        # Phase 3-tuned forex params (CLAUDE_NOTES.txt 2026-08-02/03).
        "strategies": {
            "VWAP Mean Reversion": {"entry_deviation_pct": 0.2},
            "Bollinger Mean Reversion": {"period": 20, "num_std": 4.0},
        },
    },
]

RESULTS_PATH = Path(__file__).resolve().parent / "risk_dial_sweep_results.jsonl"


def main() -> None:
    client = PolygonClient()
    total_runs = sum(len(b["strategies"]) for b in BUCKETS) * len(VOL_TARGET_GRID) * len(WINDOWS)
    done = 0
    t_start = time.monotonic()

    with RESULTS_PATH.open("a", encoding="utf-8") as out:
        for bucket in BUCKETS:
            for strategy_name, params in bucket["strategies"].items():
                for target_vol in VOL_TARGET_GRID:
                    for from_date, to_date in WINDOWS:
                        done += 1
                        elapsed = time.monotonic() - t_start
                        print(
                            f"[{elapsed:7.1f}s] ({done}/{total_runs}) {bucket['label']} / "
                            f"{strategy_name} / target_vol_ann={target_vol} / {from_date}..{to_date}",
                            flush=True,
                        )
                        rows = run_scan(
                            tickers=bucket["tickers"],
                            strategy_names=[strategy_name],
                            from_date=from_date,
                            to_date=to_date,
                            client=client,
                            strategy_params={strategy_name: params},
                            vol_target_enabled=True,
                            target_vol_ann=target_vol,
                            market_calendar=bucket["market_calendar"],
                        )
                        aggs = aggregate_by_strategy(rows)
                        agg = aggs[0] if aggs else None
                        record = {
                            "bucket": bucket["label"],
                            "strategy": strategy_name,
                            "params": params,
                            "target_vol_ann": target_vol,
                            "from_date": from_date,
                            "to_date": to_date,
                            "mean_sharpe": agg.mean_sharpe if agg else None,
                            "mean_return": agg.mean_return if agg else None,
                            "pct_profitable": agg.pct_profitable if agg else None,
                            "total_trades": agg.total_trades if agg else 0,
                            "num_errors": agg.num_errors if agg else len(bucket["tickers"]),
                        }
                        out.write(json.dumps(record) + "\n")
                        out.flush()

    print(f"Done. {total_runs} scans in {time.monotonic() - t_start:.1f}s. Results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
