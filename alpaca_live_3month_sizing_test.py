"""3-month historical replay of AlpacaLive's real starting balance against the
current live roster at the current sizing rate (2026-08-15).

Fetches AlpacaLive's REAL current equity via a read-only broker snapshot call
(no orders placed), then re-runs the 6 currently-ACTIVE roster combos over
the trailing 3 months as if each one, alone, traded that real starting
balance at today's effective sizing rate (dynamic_size_fn=0.50 - AlpacaLive
uses the sliding-scale override, but at $53 equity it long ago flattened past
e_hi=$2 to the flat 50% floor rate, so a constant 0.50 fraction is a faithful
proxy, not an approximation of a different mechanism).

CAVEAT stated here and in the printed output: this is 6 SEPARATE single-
ticker backtests, each with the account's full real balance as its own
starting cash - same "single-ticker-in-isolation" limitation as every other
sizing sweep in this project (see risk_dial_sizing_sweep.py). It answers "if
this one combo alone had the whole account," not "what does the real account
do with all 6 trading concurrently off one shared cash pool" - the engine has
no joint multi-ticker portfolio mode, and building an allocation assumption
to fake one would be inventing precision this doesn't have. Historical/
backtested only, not a forecast.

Run: python alpaca_live_3month_sizing_test.py
"""

from __future__ import annotations

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
SIZING_FRACTION = 0.50  # today's control.json sizing_value (Aggressive) / AlpacaLive's flattened slide rate
SLIPPAGE_BPS = 2.0


def main() -> int:
    accounts = build_broker_accounts([ALPACA_LIVE_ID])
    if not accounts:
        print("AlpacaLive account not found / not linked.")
        return 1
    snapshot = accounts[0].get_account_snapshot()
    starting_cash = snapshot.equity
    print(f"AlpacaLive REAL current equity (fetched now): ${starting_cash:.2f} "
          f"(cash ${snapshot.cash:.2f})\n")

    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=90)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (trailing ~3 months)")
    print(f"Sizing fraction tested: {SIZING_FRACTION * 100:.0f}% of equity per trade\n")

    combos = [(e.ticker, e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"]
    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    print(f"{'ticker/strategy':<38}{'end $':>10}{'return%':>10}{'maxDD%':>9}{'worst $':>10}{'trades':>8}{'win%':>7}")
    print("-" * 92)
    total_end = 0.0
    for ticker, strategy_name, params in combos:
        label = f"{ticker}/{strategy_name}"
        try:
            bars = client.get_aggregates(ticker, from_date.isoformat(), to_date.isoformat(), 1, "minute")
        except Exception as e:  # noqa: BLE001
            print(f"{label:<38} fetch failed: {e}")
            continue
        if bars.empty or len(bars) < 100:
            print(f"{label:<38} too few bars, skipped")
            continue

        strategy = build_strategy(strategy_name, params)
        engine = BacktestEngine(
            starting_cash=starting_cash, slippage_bps=SLIPPAGE_BPS,
            dynamic_size_fn=lambda cash: SIZING_FRACTION,
        )
        result = engine.run(bars, strategy)
        try:
            report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
        except ValueError:
            print(f"{label:<38} not enough data for metrics")
            continue

        end_equity = result.equity_curve.iloc[-1]
        closed = [t for t in result.trades if t.pnl is not None]
        worst = min((t.pnl for t in closed), default=0.0)
        total_end += end_equity - starting_cash  # incremental $ if summed naively across combos (see caveat)

        print(f"{label:<38}{end_equity:>10.2f}{report.total_return * 100:>10.2f}"
              f"{report.max_drawdown * 100:>9.2f}{worst:>10.2f}{report.num_trades:>8}"
              f"{report.win_rate * 100:>6.0f}%")

    print("-" * 92)
    print(
        f"\nCAVEAT: each row is its OWN isolated run starting from the full real ${starting_cash:.2f} "
        "balance - NOT 6 combos sharing one cash pool the way the real account actually does. Do "
        "not sum the 'end $' column as if it were the account's combined result; the naive sum of "
        f"$-change across all rows is ${total_end:+.2f}, shown only to make that mis-read impossible "
        "to do quietly. Historical replay of real market data, not a forecast."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
