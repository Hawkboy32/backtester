"""Simulates the Kraken Funded $1K challenge (grow $1,000 to $1,120 before
ever dropping below $970, no time limit) against REAL historical BTC data,
sweeping position size to find what actually works.

WHY THIS IS A DIFFERENT PROBLEM than Chopper's normal strategy validation:
every other backtest in this project optimizes for long-term risk-adjusted
return (Sharpe, avoiding deep drawdowns over months). The challenge is a
one-shot race between two FIXED, ABSOLUTE thresholds measured from a single
starting point - not a rolling/trailing drawdown, not a return rate. Size
too aggressively and a single bad trade can blow the 3% floor before you've
gone anywhere; size too conservatively and you may never realistically
reach +12%. Position size is the one lever that trades these off against
each other - so that's what this sweeps, holding the strategy itself
(VWAP Mean Reversion, the one with a real live track record on KRKAPI)
fixed.

METHODOLOGY: walk-forward in spirit, not a single lucky window - runs the
SAME sizing level from many different real historical start dates (every
2 weeks, across ~2 years of data) and reports the ACTUAL PASS RATE across
all of them, not just one outcome. A sizing level that passes 90% of the
time from one cherry-picked date but 20% of the time in general is not
"90% likely to work" - this is built specifically to catch that.

RUN IN CHUNKS, NOT ONE BLOCKING CALL (2026-09-27): the engine is a bar-by-bar
Python loop (~1.5ms/bar measured), so the full 9-size-level sweep is a
multi-hour job. A single `python kraken_funded_challenge_sim.py` invocation
covering all 9 levels proved fragile in practice - it got silently
suspended for 8+ hours overnight (zero accumulated CPU time, confirmed via
Get-Process) with no way to tell "still working" from "stuck" apart from a
handful of print statements. So this now fetches bars ONCE into a local
cache, then each size level runs as its own short, independent process that
appends one row to a results CSV - if one run dies or the machine sleeps
mid-chunk, only that chunk is lost, not the whole sweep, and progress is
visible between chunks rather than only at the very end.

SIZING RESULT (2026-09-27): VWAP Mean Reversion topped out at a 9.4% real pass
rate (75% sizing) - not a viable challenge strategy at any size tested. Rather
than assume sizing was the only lever, added a SECOND sweep axis: strategy
choice, holding sizing fixed at that same best-found level (BEST_SIZE_SO_FAR),
across a shortlist of already-implemented, already-registered strategies
(STRATEGY_CANDIDATES) that either have a real live track record on crypto
tickers or were sourced from browsing alphainsider.com's crypto-tagged
strategies for ideas worth trying (divergence, MACD-based - see
[[project_trading_bot_alphainsider_strategies]] for why AlphaInsider is
treated as an idea source, not a code source, and the verification discipline
that implies - these are all standard, already-verified indicator
implementations already in STRATEGY_REGISTRY, not new unverified code).

Usage:
    .venv/Scripts/python kraken_funded_challenge_sim.py --fetch
    .venv/Scripts/python kraken_funded_challenge_sim.py --size 0.05
    .venv/Scripts/python kraken_funded_challenge_sim.py --size 0.10
    ... (one --size call per SIZE_LEVELS entry) ...
    .venv/Scripts/python kraken_funded_challenge_sim.py --report
    .venv/Scripts/python kraken_funded_challenge_sim.py --strategy "RSI Divergence"
    ... (one --strategy call per STRATEGY_CANDIDATES entry) ...
    .venv/Scripts/python kraken_funded_challenge_sim.py --strategy-report
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import pandas as pd  # noqa: E402

from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

TICKER = "X:BTCUSD"
STRATEGY_NAME = "VWAP Mean Reversion"
TIMESPAN = "minute"
MULTIPLIER = 1

START_BALANCE = 1000.0
PASS_TARGET = 1120.0   # +12%
FAIL_FLOOR = 970.0     # -3%, FIXED at the start balance, never moves

# Position sizes to sweep - fraction of CURRENT equity spent on each new
# entry (compounds as the balance grows/shrinks, same as the real challenge
# balance would).
SIZE_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00]

# Best real pass rate from the SIZE_LEVELS sweep (75%, 9.4% - see module
# docstring) - held fixed while sweeping strategy choice instead, so the two
# sweeps aren't conflated (a strategy that's actually better could still lose
# to VWAP Mean Reversion if compared at a sizing that doesn't suit it).
BEST_SIZE_SO_FAR = 0.75

# Already-implemented, already-registered strategies worth testing against
# THIS specific problem (a fixed-target race on crypto, not long-term
# risk-adjusted return) - not new code, see module docstring for sourcing.
STRATEGY_CANDIDATES = [
    "RSI Divergence",
    "MACD Crossover",
    "Momentum ROC",
    "Linear Regression Channel",
    "DMI/ADX Trend",
    "DMI/DPO Guard",
    "EMA/RSI Confirmation",
    "Triple EMA Ribbon",
    "Bollinger Squeeze Breakout",
]

# Best strategy from the STRATEGY_CANDIDATES sweep (VWAP Mean Reversion
# remained the best of all 10 tested, 9.4% - see kraken_sim_strategy_results.csv
# and CLAUDE_NOTES.txt's 2026-09-27 entry) - held fixed while sweeping ticker
# choice instead. Every ticker here is from data/crypto_universe.csv, already
# confirmed live-tradeable on Kraken (KrakenBroker._pair(), verified against
# a real AssetPairs query 2026-08-19, not guessed) - X:BTCUSD excluded since
# it's the one already fully tested above.
BEST_STRATEGY_SO_FAR = "VWAP Mean Reversion"
TICKER_CANDIDATES = [
    "X:ETHUSD", "X:SOLUSD", "X:XRPUSD", "X:ADAUSD", "X:DOGEUSD", "X:AVAXUSD",
    "X:LINKUSD", "X:DOTUSD", "X:LTCUSD", "X:BCHUSD", "X:UNIUSD", "X:ATOMUSD",
    "X:XLMUSD", "X:ETCUSD",
]

# How far back to pull data, and how often to start a fresh simulated
# challenge attempt within that window.
HISTORY_DAYS = 730  # ~2 years, matching this project's usual validation window
START_SPACING_DAYS = 14  # a new simulated attempt every 2 weeks of history

# Caps how much data each simulated attempt actually walks through. Without
# this, the engine keeps running every window all the way to the END of the
# 2-year series even after PASS/FAIL is already decided. A real challenge
# has no time limit, but any attempt that hasn't resolved within two months
# isn't telling us anything useful about "does this sizing work" either way,
# so it's cheap and safe to call it INCONCLUSIVE there and move on.
MAX_HORIZON_DAYS = 60

BARS_CACHE_PATH = Path(__file__).resolve().parent / "kraken_sim_bars.pkl"
RESULTS_PATH = Path(__file__).resolve().parent / "kraken_sim_results.csv"
RESULTS_FIELDS = ["size_fraction", "n", "n_pass", "n_fail", "n_inconclusive", "pass_rate", "avg_days_to_pass"]

STRATEGY_RESULTS_PATH = Path(__file__).resolve().parent / "kraken_sim_strategy_results.csv"
STRATEGY_RESULTS_FIELDS = [
    "strategy_name", "size_fraction", "n", "n_pass", "n_fail", "n_inconclusive", "pass_rate", "avg_days_to_pass",
]

TICKER_RESULTS_PATH = Path(__file__).resolve().parent / "kraken_sim_ticker_results.csv"
TICKER_RESULTS_FIELDS = [
    "ticker", "strategy_name", "size_fraction", "n", "n_pass", "n_fail", "n_inconclusive", "pass_rate", "avg_days_to_pass",
]


def _bars_cache_path(ticker: str) -> Path:
    # BTC keeps its original bare filename (kraken_sim_bars.pkl) so the
    # already-cached file from the size/strategy sweeps above is reused as-is,
    # not re-fetched under a new name.
    if ticker == TICKER:
        return BARS_CACHE_PATH
    safe = ticker.replace(":", "_")
    return Path(__file__).resolve().parent / f"kraken_sim_bars_{safe}.pkl"


def _simulate_one(bars: pd.DataFrame, size_fraction: float, strategy_name: str = STRATEGY_NAME) -> dict:
    """Runs ONE challenge attempt starting at the first bar of `bars`
    (caller already sliced the data to the desired start date AND capped it
    to MAX_HORIZON_DAYS) until the equity curve crosses PASS_TARGET, drops
    below FAIL_FLOOR, or the data runs out (INCONCLUSIVE - either genuinely
    out of historical data, or past the horizon cap)."""
    strategy = build_strategy(strategy_name)
    engine = BacktestEngine(
        starting_cash=START_BALANCE,
        commission_per_trade=0.0,
        slippage_bps=5.0,  # crypto spreads are wider than equities; conservative
        dynamic_size_fn=lambda cash: size_fraction,
    )
    result = engine.run(bars, strategy)
    equity = result.equity_curve

    passed_at = equity[equity >= PASS_TARGET]
    failed_at = equity[equity <= FAIL_FLOOR]

    pass_time = passed_at.index[0] if len(passed_at) else None
    fail_time = failed_at.index[0] if len(failed_at) else None

    if pass_time is not None and (fail_time is None or pass_time <= fail_time):
        days = (pass_time - equity.index[0]).total_seconds() / 86400
        return {"outcome": "PASS", "days": days}
    if fail_time is not None:
        days = (fail_time - equity.index[0]).total_seconds() / 86400
        return {"outcome": "FAIL", "days": days}
    return {"outcome": "INCONCLUSIVE", "days": None}


def _fetch_bars(force: bool = False, ticker: str = TICKER) -> pd.DataFrame:
    """Fetches once and caches (per-ticker path) so every --size/--strategy/
    --ticker chunk (and any restart after a kill/hang) reuses the exact same
    data instead of re-pulling ~1M bars from Polygon every single time."""
    cache_path = _bars_cache_path(ticker)
    if cache_path.exists() and not force:
        print(f"Loading cached bars from {cache_path.name}...")
        bars = pd.read_pickle(cache_path)
        print(f"{len(bars)} bars loaded.\n")
        return bars

    client = PolygonClient()
    to_date = pd.Timestamp.now(tz="UTC").date()
    from_date = to_date - pd.Timedelta(days=HISTORY_DAYS)
    print(f"Fetching {ticker} {MULTIPLIER}{TIMESPAN} bars, {from_date} to {to_date}...", flush=True)
    bars = client.get_aggregates(
        ticker=ticker, from_date=str(from_date), to_date=str(to_date),
        multiplier=MULTIPLIER, timespan=TIMESPAN,
    )
    print(f"{len(bars)} bars fetched. Caching to {cache_path.name}...\n", flush=True)
    bars.to_pickle(cache_path)
    return bars


def _start_points(bars: pd.DataFrame) -> list:
    start_points = [
        t for i, t in enumerate(bars.index)
        if i == 0 or (t - bars.index[0]).days % START_SPACING_DAYS == 0
    ]
    # de-dupe consecutive bars landing on the same day
    seen_days = set()
    dedup_starts = []
    for t in start_points:
        if t.date() not in seen_days:
            seen_days.add(t.date())
            dedup_starts.append(t)
    return dedup_starts


def _run_one_size(bars: pd.DataFrame, size_fraction: float) -> dict:
    start_points = _start_points(bars)
    print(f"{size_fraction*100:.0f}% sizing: {len(start_points)} windows", flush=True)

    outcomes = []
    for i, start in enumerate(start_points):
        window = bars.loc[start:start + pd.Timedelta(days=MAX_HORIZON_DAYS)]
        if len(window) < 30:  # not enough bars left to mean anything
            continue
        outcomes.append(_simulate_one(window, size_fraction))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(start_points)}", flush=True)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
    n_inconclusive = n - n_pass - n_fail
    pass_rate = n_pass / n if n else 0.0
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "PASS"]
    avg_days = sum(pass_days) / len(pass_days) if pass_days else None

    row = {
        "size_fraction": size_fraction, "n": n, "n_pass": n_pass, "n_fail": n_fail,
        "n_inconclusive": n_inconclusive, "pass_rate": pass_rate, "avg_days_to_pass": avg_days,
    }
    avg_days_str = f"{avg_days:.1f}" if avg_days is not None else "-"
    print(f"{size_fraction*100:>5.0f}% {n_pass:>6} {n_fail:>6} {n_inconclusive:>8}   "
          f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}", flush=True)
    return row


def _run_one_strategy(bars: pd.DataFrame, strategy_name: str, size_fraction: float = BEST_SIZE_SO_FAR) -> dict:
    start_points = _start_points(bars)
    print(f"{strategy_name} @ {size_fraction*100:.0f}% sizing: {len(start_points)} windows", flush=True)

    outcomes = []
    for i, start in enumerate(start_points):
        window = bars.loc[start:start + pd.Timedelta(days=MAX_HORIZON_DAYS)]
        if len(window) < 30:
            continue
        outcomes.append(_simulate_one(window, size_fraction, strategy_name))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(start_points)}", flush=True)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
    n_inconclusive = n - n_pass - n_fail
    pass_rate = n_pass / n if n else 0.0
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "PASS"]
    avg_days = sum(pass_days) / len(pass_days) if pass_days else None

    row = {
        "strategy_name": strategy_name, "size_fraction": size_fraction, "n": n, "n_pass": n_pass,
        "n_fail": n_fail, "n_inconclusive": n_inconclusive, "pass_rate": pass_rate, "avg_days_to_pass": avg_days,
    }
    avg_days_str = f"{avg_days:.1f}" if avg_days is not None else "-"
    print(f"{strategy_name:>28} {n_pass:>6} {n_fail:>6} {n_inconclusive:>8}   "
          f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}", flush=True)
    return row


def _run_one_ticker(
    bars: pd.DataFrame, ticker: str,
    strategy_name: str = BEST_STRATEGY_SO_FAR, size_fraction: float = BEST_SIZE_SO_FAR,
) -> dict:
    start_points = _start_points(bars)
    print(f"{ticker} - {strategy_name} @ {size_fraction*100:.0f}% sizing: {len(start_points)} windows", flush=True)

    outcomes = []
    for i, start in enumerate(start_points):
        window = bars.loc[start:start + pd.Timedelta(days=MAX_HORIZON_DAYS)]
        if len(window) < 30:
            continue
        outcomes.append(_simulate_one(window, size_fraction, strategy_name))
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(start_points)}", flush=True)

    n = len(outcomes)
    n_pass = sum(1 for o in outcomes if o["outcome"] == "PASS")
    n_fail = sum(1 for o in outcomes if o["outcome"] == "FAIL")
    n_inconclusive = n - n_pass - n_fail
    pass_rate = n_pass / n if n else 0.0
    pass_days = [o["days"] for o in outcomes if o["outcome"] == "PASS"]
    avg_days = sum(pass_days) / len(pass_days) if pass_days else None

    row = {
        "ticker": ticker, "strategy_name": strategy_name, "size_fraction": size_fraction, "n": n,
        "n_pass": n_pass, "n_fail": n_fail, "n_inconclusive": n_inconclusive,
        "pass_rate": pass_rate, "avg_days_to_pass": avg_days,
    }
    avg_days_str = f"{avg_days:.1f}" if avg_days is not None else "-"
    print(f"{ticker:>12} {n_pass:>6} {n_fail:>6} {n_inconclusive:>8}   "
          f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}", flush=True)
    return row


def _append_result(row: dict) -> None:
    is_new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _append_strategy_result(row: dict) -> None:
    is_new = not STRATEGY_RESULTS_PATH.exists()
    with open(STRATEGY_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STRATEGY_RESULTS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _append_ticker_result(row: dict) -> None:
    is_new = not TICKER_RESULTS_PATH.exists()
    with open(TICKER_RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TICKER_RESULTS_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def _report() -> None:
    if not RESULTS_PATH.exists():
        print("No results yet - run some --size chunks first.")
        return
    with open(RESULTS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Results file is empty.")
        return

    print(f"{'size':>6} {'PASS':>6} {'FAIL':>6} {'INCONCL':>8}   {'pass rate':>10}  {'avg days to pass':>17}")
    best = None
    for r in rows:
        size_fraction = float(r["size_fraction"])
        pass_rate = float(r["pass_rate"])
        avg_days = r["avg_days_to_pass"]
        avg_days_str = f"{float(avg_days):.1f}" if avg_days not in (None, "", "None") else "-"
        print(f"{size_fraction*100:>5.0f}% {r['n_pass']:>6} {r['n_fail']:>6} {r['n_inconclusive']:>8}   "
              f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}")
        if best is None or pass_rate > float(best["pass_rate"]):
            best = r

    missing = sorted(set(SIZE_LEVELS) - {float(r["size_fraction"]) for r in rows})
    if missing:
        print(f"\nStill missing: {', '.join(f'{m*100:.0f}%' for m in missing)}")
    if best is not None:
        avg_days = best["avg_days_to_pass"]
        suffix = f" (avg {float(avg_days):.1f} days when it did)" if avg_days not in (None, "", "None") else ""
        print(f"\nBest real pass rate so far: {float(best['size_fraction'])*100:.0f}% sizing -> "
              f"{float(best['pass_rate'])*100:.1f}% of {best['n']} historical attempts passed{suffix}.")


def _strategy_report() -> None:
    if not STRATEGY_RESULTS_PATH.exists():
        print("No strategy results yet - run some --strategy chunks first.")
        return
    with open(STRATEGY_RESULTS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Strategy results file is empty.")
        return

    print(f"{'strategy':>28} {'PASS':>6} {'FAIL':>6} {'INCONCL':>8}   {'pass rate':>10}  {'avg days to pass':>17}")
    best = None
    for r in rows:
        pass_rate = float(r["pass_rate"])
        avg_days = r["avg_days_to_pass"]
        avg_days_str = f"{float(avg_days):.1f}" if avg_days not in (None, "", "None") else "-"
        print(f"{r['strategy_name']:>28} {r['n_pass']:>6} {r['n_fail']:>6} {r['n_inconclusive']:>8}   "
              f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}")
        if best is None or pass_rate > float(best["pass_rate"]):
            best = r

    missing = sorted(set(STRATEGY_CANDIDATES) - {r["strategy_name"] for r in rows})
    if missing:
        print(f"\nStill missing: {', '.join(missing)}")
    if best is not None:
        avg_days = best["avg_days_to_pass"]
        suffix = f" (avg {float(avg_days):.1f} days when it did)" if avg_days not in (None, "", "None") else ""
        print(f"\nBest real pass rate so far: {best['strategy_name']} @ "
              f"{float(best['size_fraction'])*100:.0f}% sizing -> "
              f"{float(best['pass_rate'])*100:.1f}% of {best['n']} historical attempts passed{suffix}.")


def _ticker_report() -> None:
    if not TICKER_RESULTS_PATH.exists():
        print("No ticker results yet - run some --ticker chunks first.")
        return
    with open(TICKER_RESULTS_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Ticker results file is empty.")
        return

    print(f"{'ticker':>12} {'PASS':>6} {'FAIL':>6} {'INCONCL':>8}   {'pass rate':>10}  {'avg days to pass':>17}")
    best = None
    for r in rows:
        pass_rate = float(r["pass_rate"])
        avg_days = r["avg_days_to_pass"]
        avg_days_str = f"{float(avg_days):.1f}" if avg_days not in (None, "", "None") else "-"
        print(f"{r['ticker']:>12} {r['n_pass']:>6} {r['n_fail']:>6} {r['n_inconclusive']:>8}   "
              f"{pass_rate*100:>9.1f}%  {avg_days_str:>17}")
        if best is None or pass_rate > float(best["pass_rate"]):
            best = r

    missing = sorted(set(TICKER_CANDIDATES) - {r["ticker"] for r in rows})
    if missing:
        print(f"\nStill missing: {', '.join(missing)}")
    if best is not None:
        avg_days = best["avg_days_to_pass"]
        suffix = f" (avg {float(avg_days):.1f} days when it did)" if avg_days not in (None, "", "None") else ""
        print(f"\nBest real pass rate so far: {best['ticker']} ({best['strategy_name']} @ "
              f"{float(best['size_fraction'])*100:.0f}% sizing) -> "
              f"{float(best['pass_rate'])*100:.1f}% of {best['n']} historical attempts passed{suffix}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="Fetch and cache bars, then exit.")
    parser.add_argument("--refetch", action="store_true", help="Force a fresh fetch even if cached.")
    parser.add_argument("--size", type=float, default=None, help="Run just this one size fraction (e.g. 0.05).")
    parser.add_argument("--report", action="store_true", help="Print results collected so far, then exit.")
    parser.add_argument(
        "--strategy", type=str, default=None,
        help=f"Run just this one strategy at BEST_SIZE_SO_FAR ({BEST_SIZE_SO_FAR*100:.0f}%% sizing).",
    )
    parser.add_argument("--strategy-report", action="store_true", help="Print strategy results so far, then exit.")
    parser.add_argument(
        "--ticker", type=str, default=None,
        help="Run just this one ticker (e.g. X:ETHUSD) at BEST_STRATEGY_SO_FAR/BEST_SIZE_SO_FAR.",
    )
    parser.add_argument("--ticker-report", action="store_true", help="Print ticker results so far, then exit.")
    args = parser.parse_args()

    if args.report:
        _report()
        return
    if args.strategy_report:
        _strategy_report()
        return
    if args.ticker_report:
        _ticker_report()
        return

    if args.ticker is not None:
        bars = _fetch_bars(force=args.refetch, ticker=args.ticker)
        if args.fetch:
            return
        row = _run_one_ticker(bars, args.ticker)
        _append_ticker_result(row)
        return

    bars = _fetch_bars(force=args.refetch)
    if args.fetch:
        return

    if args.strategy is not None:
        row = _run_one_strategy(bars, args.strategy)
        _append_strategy_result(row)
        return

    if args.size is not None:
        row = _run_one_size(bars, args.size)
        _append_result(row)
        return

    # No flags: run every size level in-process, one after another (the
    # original all-in-one behavior - kept for convenience on a short data
    # range, but see the module docstring for why the full 2-year sweep
    # should be driven via separate --size chunks instead).
    for size_fraction in SIZE_LEVELS:
        row = _run_one_size(bars, size_fraction)
        _append_result(row)
    _report()


if __name__ == "__main__":
    main()
