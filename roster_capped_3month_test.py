"""3-month validation of the per-ticker liquidity caps just wired into
auto_trader.py (2026-08-16) - all 6 active roster combos, at each of the
three risk tiers (Conservative/Moderate/Aggressive), liquidity-aware
slippage throughout, comparing what each tier WOULD have done before the
cap existed (global sizing_value applied directly) against what it does now
(clamped through risk_presets.ticker_sizing_cap - the actual live function,
not a re-implementation, so this measures the real deployed logic).

No deposits - isolates the sizing-cap question the same way
roster_liquidity_efficient_sizing.py did, from a different one (the deposit
mechanics were already validated separately).

Under FLAT slippage this comparison would show nothing (risk_dial_sizing_
sweep.py already proved Sharpe/return scale linearly with fraction there) -
the liquidity model is what makes "capped vs uncapped" a real question with
a real answer instead of a no-op.

Run: python roster_capped_3month_test.py
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
from backtester.risk_presets import RISK_PRESETS, ticker_sizing_cap  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

STARTING_CASH = 53.20  # AlpacaLive's real balance, consistent with the rest of this session's work
SLIPPAGE_BPS = 2.0
BASE_BPS, IMPACT_BPS_PER_PCT, CAP_BPS = 2.0, 8.0, 500.0
TIERS = ["Conservative", "Moderate", "Aggressive"]


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
    from_date = to_date - timedelta(days=90)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (~3 months, no deposits)")
    print(f"Starting cash: ${STARTING_CASH:.2f} (AlpacaLive real balance)")
    print(f"Liquidity model: {BASE_BPS}bps base + {IMPACT_BPS_PER_PCT}bps per 1% of bar volume, "
          f"capped {CAP_BPS}bps\n")

    combos = [(e.ticker, e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"]
    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    cache: dict[str, tuple] = {}
    for ticker, strategy_name, params in combos:
        print(f"Fetching {ticker}...")
        bars = _fetch_with_retry(client, ticker, from_date.isoformat(), to_date.isoformat())
        rec = _Rec(build_strategy(strategy_name, params))
        BacktestEngine(starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS).run(bars, rec)
        cache[ticker] = (bars, rec.signals)
    print()

    total_uncapped_by_tier = {t: 0.0 for t in TIERS}
    total_capped_by_tier = {t: 0.0 for t in TIERS}

    all_curves: dict[str, dict[str, dict[str, list]]] = {}
    tier_caps: dict[str, dict[str, float]] = {}

    for tier in TIERS:
        global_pct = RISK_PRESETS[tier]["sizing_value"]
        all_curves[tier] = {}
        tier_caps[tier] = {}
        print(f"=== {tier} (global {global_pct:.0f}%) ===")
        print(f"{'ticker/strategy':<32}{'used %':>8}{'capped %':>10}{'uncapped $':>12}{'capped $':>12}{'diff':>10}")
        print("-" * 84)
        for ticker, strategy_name, params in combos:
            bars, signals = cache[ticker]
            cap = ticker_sizing_cap(ticker, tier)
            capped_pct = min(global_pct, cap) if cap is not None else global_pct

            uncapped_engine = BacktestEngine(
                starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                dynamic_size_fn=lambda cash, f=global_pct / 100: f,
                variable_slippage_fn=liquidity_slippage,
            )
            uncapped_result = uncapped_engine.run(bars, _Replay(list(signals)))

            capped_engine = BacktestEngine(
                starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
                dynamic_size_fn=lambda cash, f=capped_pct / 100: f,
                variable_slippage_fn=liquidity_slippage,
            )
            capped_result = capped_engine.run(bars, _Replay(list(signals)))

            uncapped_end = uncapped_result.equity_curve.iloc[-1]
            capped_end = capped_result.equity_curve.iloc[-1]
            total_uncapped_by_tier[tier] += uncapped_end
            total_capped_by_tier[tier] += capped_end

            label = f"{ticker}/{strategy_name}"
            marker = " *" if capped_pct < global_pct else ""
            print(f"{label:<32}{global_pct:>7.0f}%{capped_pct:>9.0f}%{uncapped_end:>12.2f}"
                  f"{capped_end:>12.2f}{capped_end - uncapped_end:>+10.2f}{marker}")

            uncapped_daily = uncapped_result.equity_curve.resample("1D").last().dropna()
            capped_daily = capped_result.equity_curve.resample("1D").last().dropna()
            all_curves[tier][label] = {
                "uncapped": [[ts.strftime("%Y-%m-%d"), round(float(v), 4)] for ts, v in uncapped_daily.items()],
                "capped": [[ts.strftime("%Y-%m-%d"), round(float(v), 4)] for ts, v in capped_daily.items()],
            }
            tier_caps[tier][label] = {"global_pct": global_pct, "capped_pct": capped_pct}
        print()

    print("=== SUMMARY: sum of all 6 combos' end balances, per tier ===")
    print(f"{'tier':<14}{'uncapped total $':>18}{'capped total $':>16}{'difference':>14}")
    print("-" * 62)
    for tier in TIERS:
        u, c = total_uncapped_by_tier[tier], total_capped_by_tier[tier]
        print(f"{tier:<14}{u:>18,.2f}{c:>16,.2f}{c - u:>+14,.2f}")
    print("\n* = this ticker's cap actually reduced sizing below the tier's global rate")
    print("(Remember: each row is its own isolated run, not a shared cash pool - see the earlier caveat.)")

    out_path = Path(__file__).resolve().parent / "roster_capped_3month_curves.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({
            "starting_cash": STARTING_CASH, "from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            "tiers": TIERS, "tier_caps": tier_caps, "curves": all_curves,
        }, f)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
