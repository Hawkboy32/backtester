"""Same 3-month AlpacaLive replay as alpaca_live_3month_sizing_test.py, but:
  - three sizing fractions (5% / 20% / 100%) instead of one, so the shape of
    the risk/reward tradeoff is visible across the widened dial range agreed
    2026-08-15, not just at one point on it.
  - a $40 deposit added each month during the window, via BacktestEngine's
    new `deposits` param (added today specifically for this - see engine.py's
    own docstring on it, and the regression check in verify_protective_exits.py
    still passing 16/16 after the change).

DEPOSIT DATES: the natural "1st of the month" for Jun/Jul/Aug 2026 lands on a
Saturday for August - a raw calendar date would silently never fire, since
the engine only applies a deposit on a date that actually has a bar. Instead,
each deposit is snapped to the first REAL trading session on or after that
calendar date, computed once from a reference ticker's actual fetched bars
(all 6 combos are US equities sharing one NYSE calendar) and reused
identically across every combo/fraction so all 18 runs deposit on the exact
same real dates.

Same "6 separate isolated single-ticker runs" caveat as the first version -
still not a joint shared-cash-pool simulation of the real account.

Run: python alpaca_live_3month_deposits_test.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import roster  # noqa: E402
from backtester.accounts import build_broker_accounts  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402

ALPACA_LIVE_ID = "25b452ff-28ac-4a15-84a5-79be8185a437"
FRACTIONS = [0.05, 0.20, 1.00]
MONTHLY_DEPOSIT = 40.0
SLIPPAGE_BPS = 2.0
OUT_JSON = Path(__file__).resolve().parent / "alpaca_3month_deposits_curves.json"


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
    from_date = to_date - timedelta(days=90)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()}")
    print(f"Fractions tested: {[f'{f*100:.0f}%' for f in FRACTIONS]}")
    print(f"Monthly deposit: ${MONTHLY_DEPOSIT:.2f}\n")

    combos = [(e.ticker, e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"]
    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    # Reference session calendar (from the first combo's ticker) -> real
    # deposit dates, reused identically for every run below.
    ref_ticker = combos[0][0]
    ref_bars = client.get_aggregates(ref_ticker, from_date.isoformat(), to_date.isoformat(), 1, "minute")
    session_dates = sorted({ts.date() for ts in ref_bars.index})
    month_starts = [date(2026, 6, 1), date(2026, 7, 1), date(2026, 8, 1)]
    deposit_dates = [d for d in (_snap_to_session(m, session_dates) for m in month_starts) if d is not None]
    deposits = {d: MONTHLY_DEPOSIT for d in deposit_dates}
    print(f"Deposit dates (snapped to real trading sessions): {[d.isoformat() for d in deposit_dates]}")
    total_contributed = starting_cash + len(deposit_dates) * MONTHLY_DEPOSIT
    print(f"Total contributed by end of window: ${total_contributed:.2f} "
          f"(${starting_cash:.2f} start + {len(deposit_dates)} x ${MONTHLY_DEPOSIT:.2f})\n")

    all_curves: dict[str, dict[str, list]] = {}
    summary_rows = []

    for fraction in FRACTIONS:
        frac_key = f"{fraction * 100:.0f}%"
        all_curves[frac_key] = {}
        print(f"=== {frac_key} sizing ===")
        print(f"{'ticker/strategy':<38}{'end $':>10}{'contributed $':>14}{'gain/loss $':>12}{'maxDD%':>9}{'trades':>8}")
        print("-" * 91)
        for ticker, strategy_name, params in combos:
            label = f"{ticker}/{strategy_name}"
            bars = client.get_aggregates(ticker, from_date.isoformat(), to_date.isoformat(), 1, "minute")
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
                print(f"{label:<38} not enough data")
                continue

            end_equity = result.equity_curve.iloc[-1]
            gain = end_equity - total_contributed
            daily = result.equity_curve.resample("1D").last().dropna()
            all_curves[frac_key][label] = [[ts.strftime("%Y-%m-%d"), round(float(v), 4)] for ts, v in daily.items()]

            print(f"{label:<38}{end_equity:>10.2f}{total_contributed:>14.2f}{gain:>+12.2f}"
                  f"{report.max_drawdown * 100:>9.2f}{report.num_trades:>8}")
            summary_rows.append({
                "fraction": frac_key, "combo": label, "end_equity": round(end_equity, 2),
                "gain_over_contributed": round(gain, 2), "max_drawdown_pct": round(report.max_drawdown * 100, 2),
                "num_trades": report.num_trades,
            })
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
