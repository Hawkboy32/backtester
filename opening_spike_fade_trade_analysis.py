"""Per-session trade-level analysis for the Opening Spike Fade idea -
rather than one aggregate Sharpe per parameter set, computes EVERY session's
opening move (first 5 min, the best opening_minutes found by tuning) and
what fading it would have done, unfiltered by any min_move_pct threshold
(filtering first would throw away exactly the data needed to look for a
pattern in). Records per-session: day-of-week, direction (up-open/down-
open), move size, exit reason (reverted-to-open vs timed-out), P&L, and
holding time - then looks for what actually separates winners from losers,
rather than assuming a single global threshold is the right lens.

Run: python opening_spike_fade_trade_analysis.py
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.strategies.indicators import session_dates  # noqa: E402

TICKER = "I:NDX"
OPENING_MINUTES = 5  # best single param from the tuning pass
REVERSAL_WINDOW_MINUTES = 120
START = date(2024, 6, 1)


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
    print(f"Fetching {TICKER}, {START.isoformat()} .. {end.isoformat()}...", flush=True)
    client = PolygonClient()
    bars = _fetch_with_retry(client, TICKER, START.isoformat(), end.isoformat())
    if bars is None or bars.empty:
        print("No bars.")
        return 1
    print(f"Got {len(bars)} bars\n", flush=True)

    dates = session_dates(bars.index)
    records = []
    for session_date in pd.unique(dates):
        day_bars = bars[dates == session_date]
        if len(day_bars) < 10:
            continue
        session_start = day_bars.index[0]
        session_open = day_bars["open"].iloc[0]
        if session_open <= 0:
            continue
        or_end = session_start + pd.Timedelta(minutes=OPENING_MINUTES)
        reversal_end = session_start + pd.Timedelta(minutes=REVERSAL_WINDOW_MINUTES)

        opening_window = day_bars[day_bars.index < or_end]
        post_or = day_bars[day_bars.index >= or_end]
        if opening_window.empty or post_or.empty:
            continue

        opening_move_pct = (opening_window["close"].iloc[-1] - session_open) / session_open * 100
        direction = "down-open" if opening_move_pct < 0 else "up-open"
        is_long_fade = opening_move_pct < 0

        entry_price = post_or["close"].iloc[0]
        entry_time = post_or.index[0]
        window_bars = post_or[post_or.index <= reversal_end]

        exit_price, exit_time, exit_reason = None, None, None
        for ts, row in window_bars.iloc[1:].iterrows():
            reverted = row["close"] >= session_open if is_long_fade else row["close"] <= session_open
            if reverted:
                exit_price, exit_time, exit_reason = row["close"], ts, "reverted"
                break
        if exit_price is None:
            exit_price = window_bars["close"].iloc[-1]
            exit_time = window_bars.index[-1]
            exit_reason = "timed_out"

        pnl_pct = (exit_price - entry_price) / entry_price * 100 * (1 if is_long_fade else -1)
        holding_minutes = (exit_time - entry_time).total_seconds() / 60

        records.append({
            "date": session_date, "day_of_week": session_start.strftime("%A"),
            "direction": direction, "opening_move_pct": opening_move_pct,
            "abs_move_pct": abs(opening_move_pct), "pnl_pct": pnl_pct,
            "exit_reason": exit_reason, "holding_minutes": holding_minutes,
        })

    df = pd.DataFrame(records)
    print(f"Total sessions analyzed: {len(df)}\n")

    print("=== By day of week ===")
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    by_dow = df.groupby("day_of_week").agg(
        n=("pnl_pct", "size"), win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        mean_pnl=("pnl_pct", "mean"), total_pnl=("pnl_pct", "sum"),
    ).reindex(dow_order)
    print(by_dow.to_string(float_format="%.3f"))

    print("\n=== By direction (up-open vs down-open) ===")
    by_dir = df.groupby("direction").agg(
        n=("pnl_pct", "size"), win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        mean_pnl=("pnl_pct", "mean"), total_pnl=("pnl_pct", "sum"),
    )
    print(by_dir.to_string(float_format="%.3f"))

    print("\n=== By move-size quartile (abs opening move %) ===")
    df["move_quartile"] = pd.qcut(df["abs_move_pct"], 4, labels=["Q1 smallest", "Q2", "Q3", "Q4 largest"])
    by_q = df.groupby("move_quartile", observed=True).agg(
        n=("pnl_pct", "size"), win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        mean_pnl=("pnl_pct", "mean"), mean_abs_move=("abs_move_pct", "mean"),
    )
    print(by_q.to_string(float_format="%.3f"))

    print("\n=== By exit reason ===")
    by_exit = df.groupby("exit_reason").agg(
        n=("pnl_pct", "size"), win_rate=("pnl_pct", lambda s: (s > 0).mean() * 100),
        mean_pnl=("pnl_pct", "mean"),
    )
    print(by_exit.to_string(float_format="%.3f"))

    print(f"\nCorrelation(abs_move_pct, pnl_pct): {df['abs_move_pct'].corr(df['pnl_pct']):.3f}")
    print(f"Correlation(holding_minutes, pnl_pct): {df['holding_minutes'].corr(df['pnl_pct']):.3f}")

    df.to_csv(Path(__file__).resolve().parent / "opening_spike_fade_trades.csv", index=False)
    print(f"\nSaved per-session detail: opening_spike_fade_trades.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
