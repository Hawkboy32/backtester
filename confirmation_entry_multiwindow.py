"""Multi-window validation for confirm_turn_bars (see
woolly-zooming-kurzweil.md) - the user's own idea: instead of entering the
instant VWAP/Bollinger mean reversion's deviation threshold is crossed, wait
for evidence of a turn back toward the mean first. Same discipline as every
other validation in this project (BTC/ETH crypto, Opening Spike Fade): a
single-window "looks better" result risks fitting that window's own noise -
this checks across independent windows before trusting it.

Reuses the exact 3 non-overlapping windows sweep_protective_exits.py already
established (Jun/Jul/Aug1-13), so results are directly comparable to that
sweep's own TRAIN/TEST split rather than inventing a second windowing scheme.

UNIVERSES pulled from the real live config (roster.py + control.extra_targets
- same source sweep_protective_exits.py's _live_combos() uses), not a
hand-maintained list:
  - equity-vwap:      every active roster (ticker, "VWAP Mean Reversion")
  - equity-bollinger: every active roster (ticker, "Bollinger Mean Reversion")
  - crypto-btc / ig-forex / oanda-forex: each extra_targets group

Each universe pools ALL its tickers' trades into ONE equity curve per
config/window (fixed $/trade sizing makes this additive - same reasoning
sweep_protective_exits.py already uses) and scores ONE Sharpe for the whole
universe, not per-ticker - isolating the confirm_turn_bars question from
which specific ticker happened to do well.

CONFIGS: today's live params unchanged (confirm_turn_bars=0) vs 1/2/3 -
mirrors the acceptance_bars={1,2,3} grid from that OTHER (rejected) delay
mechanism for direct comparability, not because 1-3 is assumed to bracket
the right value.

Run: python confirmation_entry_multiwindow.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd  # noqa: E402

from backtester import roster  # noqa: E402
from backtester.auto_trader_state import load_control  # noqa: E402
from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

# Same 3 windows sweep_protective_exits.py uses (TRAIN/TEST split), minus its
# 4th sanity window (that one exists specifically to cross-check against the
# real ledger - not needed here, this script never claims to match reality
# in absolute terms, only to compare configs against each other).
WINDOWS = [
    ("Window 1 (Jun)", "2026-06-01", "2026-06-30"),
    ("Window 2 (Jul)", "2026-07-01", "2026-07-31"),
    ("Window 3 (Aug1-13)", "2026-08-01", "2026-08-13"),
]

STARTING_CASH = 10_000.0
DOLLARS_PER_TRADE = 1_000.0
SLIPPAGE_BPS = 2.0
CONFIRM_BARS_GRID = [1, 2, 3]


def _live_universes() -> dict[str, list[tuple[str, str, dict]]]:
    """Every (ticker, strategy_name, params) grouped into the 5 universes
    that are actually tuned/deployed right now - same live config
    sweep_protective_exits.py's _live_combos() reads, just grouped by
    strategy family / extra_targets group instead of flattened."""
    universes: dict[str, list[tuple[str, str, dict]]] = {}
    state = roster.load_roster()
    for e in state.entries:
        if e.status != "active":
            continue
        key = "equity-vwap" if e.strategy_name == "VWAP Mean Reversion" else \
              "equity-bollinger" if e.strategy_name == "Bollinger Mean Reversion" else None
        if key is None:
            continue  # a roster strategy this script doesn't have a confirm_turn_bars variant for
        universes.setdefault(key, []).append((e.ticker, e.strategy_name, dict(e.params or {})))

    control = load_control()
    for group in control.extra_targets:
        label = group.get("label", "")
        strategy_name = group.get("strategy_name", "")
        params = dict(group.get("strategy_params", {}))
        key = ("crypto-btc" if "Crypto BTC" in label else
               "ig-forex" if "IG Forex" in label else
               "oanda-forex" if "OANDA Forex" in label else None)
        if key is None:
            continue  # not one of the 5 tuned universes this investigation covers (e.g. NDX)
        for ticker in group.get("tickers", []):
            universes.setdefault(key, []).append((ticker, strategy_name, params))
    return universes


_CONFIRMED_NAME = {
    "VWAP Mean Reversion": "VWAP Mean Reversion (Turn-Confirmed)",
    "Bollinger Mean Reversion": "Bollinger Mean Reversion (Turn-Confirmed)",
}


def _fetch_with_retry(client: PolygonClient, ticker: str, from_date: str, to_date: str, attempts: int = 5):
    for i in range(attempts):
        try:
            return client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except PolygonError as e:
            if i == attempts - 1:
                raise
            wait = 65
            print(f"    rate limited, waiting {wait}s (attempt {i + 1}/{attempts}): {e}", flush=True)
            time.sleep(wait)


def _pooled_report(all_trades: list, ppy: float):
    """Merges trades from every ticker in a universe into ONE equity curve
    (fixed $/trade sizing makes summing them additive) and scores it. A
    synthetic step-function curve: starting cash, then cumulative realized
    P&L at each trade's exit, in chronological order - the standard way to
    combine several fixed-size positions into one portfolio view."""
    closed = sorted((t for t in all_trades if t.pnl is not None), key=lambda t: t.exit_time)
    if not closed:
        raise ValueError("no closed trades")
    equity_vals = [STARTING_CASH]
    equity_idx = [closed[0].entry_time]
    running = STARTING_CASH
    for t in closed:
        running += t.pnl
        equity_vals.append(running)
        equity_idx.append(t.exit_time)
    equity_curve = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_idx))
    return compute_report(equity_curve, closed, periods_per_year=ppy)


def main() -> int:
    universes = _live_universes()
    if not universes:
        print("No live universes found.")
        return 1
    for name, combos in universes.items():
        print(f"  {name}: " + ", ".join(f"{t}/{s.split()[0]}" for t, s, _ in combos))
    print()

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    # bars_cache[(universe, window_label, ticker)] = DataFrame
    bars_cache: dict[tuple, pd.DataFrame] = {}
    for uni_name, combos in universes.items():
        for label, start, end in WINDOWS:
            for ticker, _, _ in combos:
                key = (uni_name, label, ticker)
                if key in bars_cache:
                    continue  # a ticker can appear in one universe only here, but be defensive
                try:
                    bars = _fetch_with_retry(client, ticker, start, end)
                except PolygonError as e:
                    print(f"  {ticker} {label}: FAILED ({e})")
                    continue
                if bars is None or bars.empty or len(bars) < 100:
                    print(f"  {ticker} {label}: too few bars, skipped")
                    continue
                bars_cache[key] = bars

    scoreboard: dict[str, dict[str, list[float]]] = {}  # [universe][config_name] = [sharpe, ...]

    for uni_name, combos in universes.items():
        strategy_name = combos[0][1]
        confirmed_name = _CONFIRMED_NAME.get(strategy_name)
        if confirmed_name is None:
            continue
        configs = [("default (confirm_turn_bars=0)", 0)] + [(f"confirm_turn_bars={n}", n) for n in CONFIRM_BARS_GRID]
        scoreboard[uni_name] = {name: [] for name, _ in configs}

        print(f"=== {uni_name} ({strategy_name}) ===")
        for label, _, _ in WINDOWS:
            print(f"  {label}:")
            for config_name, confirm_bars in configs:
                all_trades = []
                for ticker, strat_name, base_params in combos:
                    bars = bars_cache.get((uni_name, label, ticker))
                    if bars is None:
                        continue
                    params = dict(base_params)
                    params["confirm_turn_bars"] = confirm_bars
                    name_to_build = confirmed_name if confirm_bars > 0 else strat_name
                    strategy = build_strategy(name_to_build, params)
                    engine = BacktestEngine(
                        starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                        fixed_dollars_per_trade=DOLLARS_PER_TRADE,
                    )
                    result = engine.run(bars, strategy)
                    all_trades.extend(result.trades)
                try:
                    report = _pooled_report(all_trades, ppy)
                except ValueError:
                    print(f"    {config_name:28}: not enough trades to score")
                    continue
                print(f"    {config_name:28}: Sharpe={report.sharpe_ratio:>7.3f}  "
                      f"trades={report.num_trades:>4}  win_rate={report.win_rate:.0%}")
                scoreboard[uni_name][config_name].append(report.sharpe_ratio)
        print(flush=True)

    print("=" * 92)
    print("SCOREBOARD (Sharpe per window, per universe)")
    print("=" * 92)
    for uni_name, configs in scoreboard.items():
        print(f"\n{uni_name}:")
        for config_name, sharpes in configs.items():
            positive = sum(1 for s in sharpes if s > 0)
            print(f"  {config_name:28}: {[round(s, 3) for s in sharpes]} "
                  f"-> positive in {positive}/{len(sharpes)} windows" if sharpes else
                  f"  {config_name:28}: no scoreable windows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
