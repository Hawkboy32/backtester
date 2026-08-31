"""2-year version of the AlpacaLive replay - MPWR/Q/PSKY only (the three
strong VWAP MR combos), same 5%/20%/100% sizing rates, same $40/month
deposit mechanism, extended from 3 months to 2 years to check whether the
short-window extrapolation (2026-08-15 chat: "MPWR at 100% crosses $800 in
~1.6 months") actually holds up over a real, fully-computed longer backtest
rather than a 2-data-point projection.

REALISM CAVEAT, stated here because it matters more at 2 years than at 3
months: this engine models flat 2bps slippage regardless of position size.
If a fraction compounds hard enough to reach a large dollar amount, a real
account would face real slippage/liquidity effects on an order this model
cannot see - a large expected result at 100% sizing over 2 years is exactly
where that gap between backtest and reality would first start to matter, not
a reason to trust the number more.

Run: python alpaca_live_2year_test.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import roster  # noqa: E402
from backtester.accounts import build_broker_accounts  # noqa: E402
from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402


def _fetch_with_retry(client: PolygonClient, ticker: str, from_date: str, to_date: str, attempts: int = 5):
    """A 2-year minute-bar pull needs many paginated requests, easily enough
    to trip Polygon's per-minute cap even with the client's own rate limiter
    (seen in practice: MPWR's fetch alone used up the budget, and the very
    next ticker's first request got a 429). Retries with a fixed cooldown
    rather than failing the whole run over a transient, expected-under-load
    condition."""
    for i in range(attempts):
        try:
            return client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except PolygonError as e:
            if i == attempts - 1:
                raise
            wait = 65
            print(f"    rate limited fetching {ticker}, waiting {wait}s (attempt {i + 1}/{attempts}): {e}")
            time.sleep(wait)

ALPACA_LIVE_ID = "25b452ff-28ac-4a15-84a5-79be8185a437"
FRACTIONS = [0.05, 0.20, 1.00]
MONTHLY_DEPOSIT = 40.0
SLIPPAGE_BPS = 2.0
TARGET_TICKERS = {"MPWR", "Q", "PSKY"}
OUT_JSON = Path(__file__).resolve().parent / "alpaca_2year_curves.json"


def _month_starts(from_date: date, to_date: date) -> list[date]:
    out = []
    y, m = from_date.year, from_date.month
    while True:
        d = date(y, m, 1)
        if d > to_date:
            break
        if d >= from_date:
            out.append(d)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _snap_to_session(target: date, session_dates: list[date]) -> date | None:
    for d in session_dates:
        if d >= target:
            return d
    return None


def main() -> int:
    accounts = build_broker_accounts([ALPACA_LIVE_ID])
    if not accounts:
        print("AlpacaLive account not found / not linked.")
        return 1
    snapshot = accounts[0].get_account_snapshot()
    starting_cash = snapshot.equity
    print(f"AlpacaLive REAL current equity (fetched now): ${starting_cash:.2f}\n")

    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=730)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (~2 years)")
    print(f"Fractions tested: {[f'{f*100:.0f}%' for f in FRACTIONS]}")
    print(f"Monthly deposit: ${MONTHLY_DEPOSIT:.2f}\n")

    entries = {e.ticker: (e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"}
    combos = [(t, *entries[t]) for t in TARGET_TICKERS if t in entries]
    if len(combos) != len(TARGET_TICKERS):
        missing = TARGET_TICKERS - {c[0] for c in combos}
        print(f"WARNING: not active in roster right now, skipping: {missing}")

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    # Fetch each ticker's bars ONCE (previously refetched per fraction - 3x
    # more API load than needed, and a real contributor to the rate-limit
    # failure this hit the first time). Cached in memory, replayed across
    # all 3 fractions.
    bars_by_ticker: dict[str, object] = {}
    for ticker, _, _ in combos:
        print(f"Fetching {ticker} (2 years of minute bars, may take a while)...")
        bars_by_ticker[ticker] = _fetch_with_retry(client, ticker, from_date.isoformat(), to_date.isoformat())
        time.sleep(5)  # breathing room before the next ticker's own burst of paginated requests

    # Whichever ticker has the LONGEST real history, not just combos[0] -
    # TARGET_TICKERS is a set, so its iteration order isn't guaranteed
    # stable across runs. An arbitrary ticker here could silently shortchange
    # a longer-history ticker's own deposit schedule down to a shorter one's
    # (caught for real in the liquidity-model follow-up script - see its
    # own comment on this same line).
    ref_ticker = min(bars_by_ticker, key=lambda t: bars_by_ticker[t].index[0])
    session_dates = sorted({ts.date() for ts in bars_by_ticker[ref_ticker].index})
    month_starts = _month_starts(from_date, to_date)
    deposit_dates = sorted({d for d in (_snap_to_session(m, session_dates) for m in month_starts) if d is not None})
    deposits = {d: MONTHLY_DEPOSIT for d in deposit_dates}
    total_contributed = starting_cash + len(deposit_dates) * MONTHLY_DEPOSIT
    print(f"{len(deposit_dates)} deposit dates, total contributed by end: ${total_contributed:.2f}\n")

    all_curves: dict[str, dict[str, list]] = {}
    for fraction in FRACTIONS:
        frac_key = f"{fraction * 100:.0f}%"
        all_curves[frac_key] = {}
        print(f"=== {frac_key} sizing ===")
        print(f"{'ticker/strategy':<32}{'end $':>14}{'contributed $':>14}{'gain/loss $':>16}{'maxDD%':>9}{'trades':>8}")
        print("-" * 93)
        for ticker, strategy_name, params in combos:
            label = f"{ticker}/{strategy_name}"
            bars = bars_by_ticker[ticker]
            strategy = build_strategy(strategy_name, params)
            engine = BacktestEngine(
                starting_cash=starting_cash, slippage_bps=SLIPPAGE_BPS,
                dynamic_size_fn=lambda cash, f=fraction: f,
                deposits=deposits,
            )
            result = engine.run(bars, strategy)
            try:
                report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
            except ValueError:
                print(f"{label:<32} not enough data")
                continue

            end_equity = result.equity_curve.iloc[-1]
            gain = end_equity - total_contributed
            weekly = result.equity_curve.resample("7D").last().dropna()
            all_curves[frac_key][label] = [[ts.strftime("%Y-%m-%d"), round(float(v), 2)] for ts, v in weekly.items()]

            print(f"{label:<32}{end_equity:>14,.2f}{total_contributed:>14.2f}{gain:>+16,.2f}"
                  f"{report.max_drawdown * 100:>9.2f}{report.num_trades:>8}")
        print()

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "starting_cash": starting_cash, "deposit_dates": [d.isoformat() for d in deposit_dates],
            "monthly_deposit": MONTHLY_DEPOSIT, "total_contributed": total_contributed,
            "from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            "fractions": [f"{f*100:.0f}%" for f in FRACTIONS], "curves": all_curves,
        }, f)
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
