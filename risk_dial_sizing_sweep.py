"""Sweep the risk-dial's PER-TRADE SIZING axis — the one that's actually live
(control.json's sizing_value, 0.5%/1%/2% of equity for Conservative/Moderate/
Aggressive per risk_presets.py) — from 1% to 100% of equity per trade.

WHY THIS IS DIFFERENT FROM risk_dial_vol_sweep.py
That sweep varied target_vol_ann (the GARCH regime multiplier) while the
engine's BASE size stayed all-in per trade (BacktestEngine.run()'s default:
spend = cash * size_multiplier, size_multiplier=1.0 absent a storm regime).
It never actually tested the dial's real sizing_value axis, because at the
time (2026-08-05) the engine had no way to model "spend X% of current
equity" at all. dynamic_size_fn (added 2026-08-10 for the sliding-scale
sizing work) closes that gap — this sweep is the first time sizing_value
itself gets a real walk-forward test rather than being picked without one.

SCOPE, per 2026-08-15 discussion
Only the 3 (bucket, strategy) groups that showed a real edge in the
vol-target sweep: Equities VWAP MR, Equities Bollinger MR, Forex Bollinger
MR. Sizing amplifies an existing edge (or its absence) — the 3 structurally
weak/negative groups from that sweep were already not real candidates
regardless of the dial, so re-testing them here would burn compute without a
decision-relevant answer.

Deliberately ISOLATED from target_vol_ann (GARCH regime sizing is OFF here,
regime_by_date=None) so this measures the sizing-fraction axis alone, not a
combination of two axes at once — same discipline as the vol sweep holding
entry params fixed while it varied target_vol_ann.

METHOD (same shortcut as sweep_protective_exits.py / sweep_tail_risk.py)
Signals are a pure function of price history (Strategy.on_bar never sees
account state), so each (ticker, strategy, window)'s signals are computed
ONCE and replayed across all 10 sizing fractions — the fetch and the
strategy computation happen once, only the cheap engine replay repeats.

REPORTED: for a compounding per-trade sizing dial, "what's the best return"
and "what's the worst-case drawdown" are different questions (see
sweep_tail_risk.py's reasoning) — both are reported per fraction, not just
mean Sharpe, since the whole point of testing up to 100% is to see where
compounding risk starts to dominate, not just where the average outcome
peaks.

Run: python risk_dial_sizing_sweep.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

WINDOWS = [
    ("2026-04-16", "2026-05-19"),
    ("2026-05-20", "2026-06-22"),
    ("2026-06-23", "2026-07-26"),
]

# Fraction of current equity spent on every new entry. Denser near the
# currently-live range (0.5-2%) where the actual decision lives, coarser
# toward 100% where the question is mainly "does compounding blow up."
SIZING_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]

STARTING_CASH = 100_000.0  # matches run_scan's/the original sweep's default
SLIPPAGE_BPS = 2.0

# Only the 3 groups with a real measured edge in risk_dial_sweep_results.jsonl.
GROUPS = [
    {
        "bucket": "Equities (TSLA/AAPL)",
        "strategy": "VWAP Mean Reversion",
        "tickers": ["TSLA", "AAPL"],
        "market_calendar": "equity",
        "params": {"entry_deviation_pct": 0.3},
    },
    {
        "bucket": "Equities (TSLA/AAPL)",
        "strategy": "Bollinger Mean Reversion",
        "tickers": ["TSLA", "AAPL"],
        "market_calendar": "equity",
        "params": {"period": 15, "num_std": 3.0},
    },
    {
        "bucket": "Forex (GBPUSD/EURGBP)",
        "strategy": "Bollinger Mean Reversion",
        "tickers": ["C:GBPUSD", "C:EURGBP"],
        "market_calendar": "forex",
        "params": {"period": 20, "num_std": 4.0},
    },
]

RESULTS_PATH = Path(__file__).resolve().parent / "risk_dial_sizing_sweep_results.jsonl"


class _Rec(Strategy):
    """Wraps a strategy and records every signal it emits, in order."""

    def __init__(self, inner):
        self.inner, self.signals = inner, []

    def on_bar(self, history, current):
        s = self.inner.on_bar(history, current)
        self.signals.append(s)
        return s

    def required_lookback(self):
        return self.inner.required_lookback()


class _Replay(Strategy):
    """Replays a pre-recorded signal sequence — lets the same signals run
    through many engine configs without recomputing the strategy each time."""

    def __init__(self, signals):
        self.signals, self.i = signals, -1

    def on_bar(self, history, current):
        self.i += 1
        return self.signals[self.i] if self.i < len(self.signals) else Signal.HOLD

    def required_lookback(self):
        return Lookback(bars=1)


def main() -> int:
    client = PolygonClient()

    # Pass 1: fetch bars + compute signals ONCE per (bucket, strategy, ticker, window).
    cache: dict[tuple[str, str, str, str], tuple] = {}
    fetch_jobs = [
        (g, ticker, from_date, to_date)
        for g in GROUPS
        for ticker in g["tickers"]
        for from_date, to_date in WINDOWS
    ]
    t_start = time.monotonic()
    for i, (g, ticker, from_date, to_date) in enumerate(fetch_jobs, 1):
        key = (g["bucket"], g["strategy"], ticker, from_date)
        print(
            f"[{time.monotonic() - t_start:7.1f}s] fetch+signals ({i}/{len(fetch_jobs)}) "
            f"{g['bucket']} / {g['strategy']} / {ticker} / {from_date}..{to_date}",
            flush=True,
        )
        try:
            bars = client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except Exception as exc:  # noqa: BLE001
            print(f"    fetch failed: {exc}")
            continue
        if bars.empty or len(bars) < 100:
            print("    too few bars, skipped")
            continue
        rec = _Rec(build_strategy(g["strategy"], g["params"]))
        BacktestEngine(starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS).run(bars, rec)
        cache[key] = (bars, rec.signals)
    print(f"\ncached {len(cache)}/{len(fetch_jobs)} (bucket, strategy, ticker, window) signal sets\n")

    # Pass 2: replay cached signals through every sizing fraction — no more API calls.
    # by_group_fraction[(bucket, strategy, fraction)] -> list of per-window dicts,
    # kept in memory so the summary table below doesn't need to re-read the JSONL.
    by_group_fraction: dict[tuple[str, str, float], list[dict]] = {}
    with RESULTS_PATH.open("a", encoding="utf-8") as out:
        for g in GROUPS:
            ppy = periods_per_year_for_calendar("minute", 1, g["market_calendar"])
            for fraction in SIZING_GRID:
                for from_date, to_date in WINDOWS:
                    sharpes, returns, drawdowns, trades_n = [], [], [], 0
                    for ticker in g["tickers"]:
                        got = cache.get((g["bucket"], g["strategy"], ticker, from_date))
                        if not got:
                            continue
                        bars, signals = got
                        engine = BacktestEngine(
                            starting_cash=STARTING_CASH,
                            slippage_bps=SLIPPAGE_BPS,
                            dynamic_size_fn=lambda cash, f=fraction: f,
                        )
                        result = engine.run(bars, _Replay(list(signals)))
                        try:
                            report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
                        except ValueError:
                            continue
                        sharpes.append(report.sharpe_ratio)
                        returns.append(report.total_return)
                        drawdowns.append(report.max_drawdown)
                        trades_n += report.num_trades

                    window_record = {
                        "mean_sharpe": statistics.mean(sharpes) if sharpes else None,
                        "mean_return": statistics.mean(returns) if returns else None,
                        "mean_max_drawdown": statistics.mean(drawdowns) if drawdowns else None,
                        "worst_max_drawdown": min(drawdowns) if drawdowns else None,
                        "total_trades": trades_n,
                        "num_tickers": len(sharpes),
                    }
                    out.write(json.dumps({
                        "bucket": g["bucket"], "strategy": g["strategy"], "sizing_fraction": fraction,
                        "from_date": from_date, "to_date": to_date, **window_record,
                    }) + "\n")
                    out.flush()
                    by_group_fraction.setdefault((g["bucket"], g["strategy"], fraction), []).append(window_record)

    print(f"Done in {time.monotonic() - t_start:.1f}s. Results: {RESULTS_PATH}\n")

    # Summary table: per (bucket, strategy, fraction), aggregated across the 3 windows.
    print(f"{'bucket / strategy':<40}{'size%':>7}{'Sharpe':>9}{'return%':>10}"
          f"{'%win folds':>12}{'meanDD%':>10}{'worstDD%':>10}")
    print("-" * 98)
    for g in GROUPS:
        label = f"{g['bucket']} / {g['strategy']}"
        for fraction in SIZING_GRID:
            windows = by_group_fraction.get((g["bucket"], g["strategy"], fraction), [])
            windows = [w for w in windows if w["mean_sharpe"] is not None]
            if not windows:
                continue
            mean_sharpe = statistics.mean(w["mean_sharpe"] for w in windows)
            mean_return = statistics.mean(w["mean_return"] for w in windows) * 100
            pct_win = sum(1 for w in windows if w["mean_return"] > 0) / len(windows) * 100
            mean_dd = statistics.mean(w["mean_max_drawdown"] for w in windows) * 100
            worst_dd = min(w["worst_max_drawdown"] for w in windows) * 100
            print(f"{label:<40}{fraction * 100:>6.0f}%{mean_sharpe:>9.2f}{mean_return:>10.2f}"
                  f"{pct_win:>11.0f}%{mean_dd:>10.2f}{worst_dd:>10.2f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
