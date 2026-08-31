"""Per-ticker liquidity-aware "efficient sizing cap" — extends the MPWR/Q/PSKY
liquidity work to every combo currently active on the real roster (PSKY, Q,
ZBRA, MPWR, BEN, BIIB), across the full 1-100% sizing grid, to answer a
different question than the earlier 3-combo runs did: not "how much does
liquidity cost at 5/20/100%" but "where does EACH ticker's own liquidity
profile stop rewarding bigger size."

WHY THIS IS A DIFFERENT QUESTION FROM THE ORIGINAL SIZING SWEEP
risk_dial_sizing_sweep.py (2026-08-15) proved Sharpe is FLAT across 1-100%
under flat slippage — scaling every trade's stake by a constant scales the
whole return series by that constant, so there's no peak to find. Liquidity
impact breaks that: it grows with participation rate, so it drags harder on
bigger trades than smaller ones. Under this model Sharpe should genuinely
peak somewhere per ticker, not stay flat — that peak (or the point where
liquidity-adjusted return starts falling behind the flat-slippage curve by a
real amount) IS the per-ticker efficient cap this script is looking for.

METHOD
For each of the 6 active combos, at each of 10 sizing fractions (same grid as
the original sizing sweep: 1/2/5/10/15/20/30/50/75/100%), TWO backtests -
flat 2bps and the same liquidity model validated 2026-08-16 (2.0bps base +
8.0bps per 1% of THAT BAR's own volume, capped 500bps) - both starting from
AlpacaLive's real $53.20, compounding, NO deposits (isolates the sizing
question from the deposit-schedule question already answered separately).
Same signals-computed-once-then-replayed shortcut as every other sweep here.

Run: python roster_liquidity_efficient_sizing.py
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
from backtester.data import PolygonClient, PolygonError  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.metrics import compute_report, periods_per_year_for_calendar  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

STARTING_CASH = 53.20  # AlpacaLive's real balance, consistent with the rest of this session's work
SIZING_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]
SLIPPAGE_BPS = 2.0
BASE_BPS, IMPACT_BPS_PER_PCT, CAP_BPS = 2.0, 8.0, 500.0
OUT_JSON = Path(__file__).resolve().parent / "roster_liquidity_efficient_sizing.json"


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
    from_date = to_date - timedelta(days=730)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (~2 years, no deposits)")
    print(f"Sizing grid: {[f'{f*100:.0f}%' for f in SIZING_GRID]}\n")

    combos = [(e.ticker, e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"]
    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    results: dict[str, list[dict]] = {}

    for ticker, strategy_name, params in combos:
        label = f"{ticker}/{strategy_name}"
        print(f"=== {label} ===")
        print(f"Fetching {ticker}...")
        bars = _fetch_with_retry(client, ticker, from_date.isoformat(), to_date.isoformat())
        if bars is None or bars.empty or len(bars) < 100:
            print("  too few bars, skipped\n")
            continue

        rec = _Rec(build_strategy(strategy_name, params))
        BacktestEngine(starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS).run(bars, rec)
        signals = rec.signals

        rows = []
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

        # Efficient cap: the fraction with the HIGHEST liquidity-adjusted Sharpe -
        # unlike the flat-slippage sweep, this should show a real peak, not a flat line.
        best = max(rows, key=lambda r: r["liq_sharpe"]) if rows else None
        # Divergence point: the smallest fraction where liquidity end-balance has
        # already fallen >20% behind the flat-slippage end-balance at that same
        # fraction - an earlier, more conservative warning than the Sharpe peak.
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
