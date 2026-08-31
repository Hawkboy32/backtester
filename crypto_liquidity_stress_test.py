"""Crypto liquidity/slippage stress test - the explicitly-flagged gap from
Phase 1 (2026-07-27, see CLAUDE_NOTES.txt): Bollinger Mean Reversion and VWAP
Mean Reversion both cleared a 3-window validation on the 15-pair crypto
universe at a flat 2bps slippage assumption "BORROWED FROM EQUITIES [...]
real crypto spreads on the less liquid pairs here (UNI, ATOM, ETC, XLM) are
plausibly wider than 2bps, and this hasn't been checked" - never done until
now, required before funding CBAPI/KRKAPI with real money.

METHOD
Reuses the exact liquidity model validated 2026-08-16 for equities
(roster_liquidity_efficient_sizing.py: 2.0bps base + 8.0bps per 1% of that
bar's own volume, capped 500bps) rather than an arbitrary flat "10-15bps"
guess - this model is asset-agnostic (pure function of shares/units vs that
bar's own volume), so it transfers to crypto without modification.

For each of the 15 crypto pairs, at each of the same 10 sizing fractions
used for the equities efficient-sizing work (1/2/5/10/15/20/30/50/75/100%),
compares flat-2bps vs the liquidity model, starting from $50 (a realistic
minimum-deposit assumption for this funding-plan exercise, not tied to a
real balance since these accounts aren't funded yet), compounding, no
deposits. Signals computed once per (ticker, strategy) then replayed across
every sizing fraction - same shortcut as every other sweep in this project.

Run: python crypto_liquidity_stress_test.py
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

from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402
from backtester.universe import load_crypto  # noqa: E402

STARTING_CASH = 50.00  # realistic minimum-deposit assumption for this exercise
SIZING_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
SLIPPAGE_BPS = 2.0
BASE_BPS, IMPACT_BPS_PER_PCT, CAP_BPS = 2.0, 8.0, 500.0
STRATEGIES = [
    ("Bollinger Mean Reversion", {}),
    ("VWAP Mean Reversion", {}),
]
OUT_JSON = Path(__file__).resolve().parent / "crypto_liquidity_stress_test.json"


def liquidity_slippage(shares: float, bar_volume: float) -> float:
    if bar_volume <= 0:
        return CAP_BPS
    return min(BASE_BPS + (100.0 * shares / bar_volume) * IMPACT_BPS_PER_PCT, CAP_BPS)


def _fetch_with_retry(client: PolygonClient, ticker: str, from_date: str, to_date: str, attempts: int = 5):
    for i in range(attempts):
        try:
            return client.get_aggregates(ticker, from_date, to_date, 1, "minute")
        except PolygonError as e:
            if i == attempts - 1:
                raise
            wait = 65
            print(f"    rate limited fetching {ticker}, waiting {wait}s (attempt {i + 1}/{attempts}): {e}")
            time.sleep(wait)


class _Rec(Strategy):
    def __init__(self, inner):
        self.inner, self.signals = inner, []

    def on_bar(self, history, current):
        s = self.inner.on_bar(history, current)
        self.signals.append(s)
        return s

    def required_lookback(self):
        return self.inner.required_lookback()


class _Replay(Strategy):
    def __init__(self, signals):
        self.signals, self.i = signals, -1

    def on_bar(self, history, current):
        self.i += 1
        return self.signals[self.i] if self.i < len(self.signals) else Signal.HOLD

    def required_lookback(self):
        return Lookback(bars=1)


def main() -> int:
    to_date = date.today() - timedelta(days=1)
    # 90 days, not 2 years - crypto trades 24/7 (~1440 bars/day vs equities'
    # ~390 over ~5/7 days), so a 2-year minute-bar window here is ~10x the
    # row count of the equivalent equities window (roster_liquidity_
    # efficient_sizing.py's basis) - closer in scale to Phase 1's own
    # 3x~33-day validation windows than to a naively-copied 2-year default.
    from_date = to_date - timedelta(days=90)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (~90 days, no deposits)", flush=True)
    print(f"Starting cash: ${STARTING_CASH:.2f}  Sizing grid: {[f'{f*100:.0f}%' for f in SIZING_GRID]}\n")

    tickers = load_crypto()["ticker"].tolist()
    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "crypto")

    results: dict[str, list[dict]] = {}

    for ticker in tickers:
        print(f"Fetching {ticker}...")
        try:
            bars = _fetch_with_retry(client, ticker, from_date.isoformat(), to_date.isoformat())
        except PolygonError as e:
            print(f"  FAILED: {e}\n")
            continue
        if bars is None or bars.empty or len(bars) < 100:
            print("  too few bars, skipped\n")
            continue

        for strategy_name, params in STRATEGIES:
            label = f"{ticker}/{strategy_name}"
            rec = _Rec(build_strategy(strategy_name, params))
            BacktestEngine(starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS).run(bars, rec)
            signals = rec.signals

            rows = []
            print(f"=== {label} ===")
            print(f"{'size':>6}{'flat $':>12}{'flat Sharpe':>13}{'liq $':>12}{'liq Sharpe':>12}{'liq/flat':>10}")
            print("-" * 65)
            for fraction in SIZING_GRID:
                flat_engine = BacktestEngine(
                    starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                    dynamic_size_fn=lambda cash, f=fraction: f,
                )
                flat_result = flat_engine.run(bars, _Replay(list(signals)))

                liq_engine = BacktestEngine(
                    starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                    dynamic_size_fn=lambda cash, f=fraction: f,
                    variable_slippage_fn=liquidity_slippage,
                )
                liq_result = liq_engine.run(bars, _Replay(list(signals)))

                try:
                    flat_report = compute_report(flat_result.equity_curve, flat_result.trades, periods_per_year=ppy)
                    liq_report = compute_report(liq_result.equity_curve, liq_result.trades, periods_per_year=ppy)
                except ValueError:
                    continue

                flat_end = flat_result.equity_curve.iloc[-1]
                liq_end = liq_result.equity_curve.iloc[-1]
                ratio = liq_end / flat_end if flat_end else 0.0
                print(f"{fraction*100:>5.0f}%{flat_end:>12,.2f}{flat_report.sharpe_ratio:>13.3f}"
                      f"{liq_end:>12,.2f}{liq_report.sharpe_ratio:>12.3f}{ratio:>10.2f}")
                rows.append({
                    "fraction": fraction, "flat_end": round(flat_end, 2), "flat_sharpe": round(flat_report.sharpe_ratio, 4),
                    "liq_end": round(liq_end, 2), "liq_sharpe": round(liq_report.sharpe_ratio, 4),
                    "liq_to_flat_ratio": round(ratio, 4), "num_trades": flat_report.num_trades,
                })
            results[label] = rows

            best = max(rows, key=lambda r: r["liq_sharpe"]) if rows else None
            divergence = next((r for r in rows if r["liq_to_flat_ratio"] < 0.80), None)
            print(f"\n  Liquidity-Sharpe peaks at {best['fraction']*100:.0f}% (Sharpe {best['liq_sharpe']:.3f})"
                  if best else "")
            print(f"  Liquidity drag exceeds 20% starting at {divergence['fraction']*100:.0f}% sizing"
                  if divergence else "  Liquidity drag never exceeds 20% in the tested range")
            print()

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({"starting_cash": STARTING_CASH, "sizing_grid": SIZING_GRID, "results": results,
                    "model": {"base_bps": BASE_BPS, "impact_bps_per_pct": IMPACT_BPS_PER_PCT, "cap_bps": CAP_BPS}}, f)
    print(f"Saved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
