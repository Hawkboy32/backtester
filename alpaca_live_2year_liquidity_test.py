"""Same 2-year MPWR/Q/PSKY replay as alpaca_live_2year_test.py, but with
liquidity/market-impact modeled instead of a flat 2bps slippage - direct
follow-up to that run's own caveat: "a position compounding into the
thousands would face real liquidity effects no backtest here can see."

THE MODEL (engine.py's new variable_slippage_fn hook, added 2026-08-16
specifically for this):
  slippage_bps = min(2.0 + participation_pct * 8.0, 500.0)
  where participation_pct = 100 * shares_traded / THIS BAR'S OWN VOLUME.

Deliberately simple and deliberately conservative, not a professionally
calibrated market-impact model - stated plainly rather than dressed up:
  - Base 2bps unchanged (spread + fees, same as every other run this project
    has done).
  - +8bps of extra impact per 1% of THIS MINUTE BAR's volume the order
    represents. Real execution algos slice a large order across many bars
    (VWAP/TWAP) rather than dumping it into one minute, so pricing impact off
    a single bar's volume overstates the cost of any order a real desk would
    actually split up - a conservative simplification, not an attempt at
    realism. It answers "how bad could this look under a pessimistic
    assumption," not "what would actually happen."
  - Capped at 500bps (5%) so a bar with near-zero volume can't produce an
    absurd fill price - stands in for what a real order would do in that
    moment (not fill, or wait), not a literal price.

Verified before trusting (see engine.py's own new checks): variable_slippage_fn
=None reproduces the OLD flat-slippage results byte-for-byte, and a synthetic
high-participation trade gets exactly the hand-calculated extra slippage.
verify_protective_exits.py still 16/16 after the engine change.

Run: python alpaca_live_2year_liquidity_test.py
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

ALPACA_LIVE_ID = "25b452ff-28ac-4a15-84a5-79be8185a437"
FRACTIONS = [0.05, 0.20, 1.00]
MONTHLY_DEPOSIT = 40.0
SLIPPAGE_BPS = 2.0
TARGET_TICKERS = {"MPWR", "Q", "PSKY"}

BASE_BPS = 2.0
# Overridable via CLI args for sensitivity checks:
#   arg 1 - impact coefficient (e.g. `4.0` for a half-strength model), since
#           8.0 was a deliberately conservative starting guess, not a
#           calibrated figure.
#   arg 2 - "hours" to also enable BacktestEngine's new regular_hours_only
#           gate (blocks entries outside 09:30-16:00 ET), added after this
#           model's own worst spikes turned out to cluster in thin
#           pre/after-market bars, not genuine large-order impact.
# Output filename tags itself with both so sensitivity runs don't overwrite
# each other or the original.
IMPACT_BPS_PER_PCT = float(sys.argv[1]) if len(sys.argv) > 1 else 8.0
REGULAR_HOURS_ONLY = len(sys.argv) > 2 and sys.argv[2] == "hours"
CAP_BPS = 500.0
_suffix = f"{IMPACT_BPS_PER_PCT:.1f}bps" + ("_hours" if REGULAR_HOURS_ONLY else "")
OUT_JSON = Path(__file__).resolve().parent / f"alpaca_2year_liquidity_curves_{_suffix}.json"


def liquidity_slippage(shares: float, bar_volume: float) -> float:
    if bar_volume <= 0:
        return CAP_BPS
    participation_pct = 100.0 * shares / bar_volume
    return min(BASE_BPS + participation_pct * IMPACT_BPS_PER_PCT, CAP_BPS)


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
    # Reuses the balance already established by the base 2-year runs earlier
    # today ($53.20) rather than calling the live broker again - this script
    # is a controlled "same setup, different model" sensitivity comparison,
    # not a fresh balance check, and a live API call here has already once
    # hung for several minutes with no output (root cause not chased down;
    # not worth a live dependency for a pure what-if run regardless).
    starting_cash = 53.20
    print(f"AlpacaLive equity (reused from earlier today's live fetch, not refetched): ${starting_cash:.2f}\n")
    print(f"Liquidity model: {BASE_BPS}bps base + {IMPACT_BPS_PER_PCT}bps per 1% of bar volume, "
          f"capped at {CAP_BPS}bps\n")

    to_date = date.today() - timedelta(days=1)
    from_date = to_date - timedelta(days=730)
    print(f"Window: {from_date.isoformat()} .. {to_date.isoformat()} (~2 years)")
    print(f"Fractions tested: {[f'{f*100:.0f}%' for f in FRACTIONS]}\n")

    entries = {e.ticker: (e.strategy_name, e.params) for e in roster.load_roster().entries if e.status == "active"}
    combos = [(t, *entries[t]) for t in TARGET_TICKERS if t in entries]

    client = PolygonClient()
    ppy = periods_per_year_for_calendar("minute", 1, "equity")

    bars_by_ticker: dict[str, object] = {}
    for ticker, _, _ in combos:
        print(f"Fetching {ticker} (cached from the earlier run if available)...")
        bars_by_ticker[ticker] = _fetch_with_retry(client, ticker, from_date.isoformat(), to_date.isoformat())
        time.sleep(2)

    # Whichever ticker has the LONGEST real history, not just combos[0] -
    # TARGET_TICKERS is a set, so its iteration order (and therefore which
    # ticker fetches first) is not guaranteed stable across process runs.
    # Picking an arbitrary ticker here silently shortchanged MPWR/PSKY's
    # deposit schedule down to whatever shorter-history ticker happened to
    # come first in a given run - caught by comparing against the earlier
    # flat-slippage run's deposit count (24) before trusting this one.
    ref_ticker = min(bars_by_ticker, key=lambda t: bars_by_ticker[t].index[0])
    session_dates = sorted({ts.date() for ts in bars_by_ticker[ref_ticker].index})
    print(f"Deposit calendar built from {ref_ticker} (longest history: {session_dates[0]} .. {session_dates[-1]})")
    month_starts = _month_starts(from_date, to_date)
    deposit_dates = sorted({d for d in (_snap_to_session(m, session_dates) for m in month_starts) if d is not None})
    deposits = {d: MONTHLY_DEPOSIT for d in deposit_dates}
    print(f"{len(deposit_dates)} deposit dates\n")

    all_curves: dict[str, dict[str, list]] = {}
    comparison_rows = []

    for fraction in FRACTIONS:
        frac_key = f"{fraction * 100:.0f}%"
        all_curves[frac_key] = {}
        print(f"=== {frac_key} sizing (liquidity-aware) ===")
        print(f"{'ticker/strategy':<32}{'end $':>14}{'trades':>8}{'avg slip bps':>14}{'max slip bps':>14}")
        print("-" * 82)
        for ticker, strategy_name, params in combos:
            label = f"{ticker}/{strategy_name}"
            bars = bars_by_ticker[ticker]
            strategy = build_strategy(strategy_name, params)

            slip_samples: list[float] = []

            def tracked_slippage(shares, bar_volume, _samples=slip_samples):
                bps = liquidity_slippage(shares, bar_volume)
                _samples.append(bps)
                return bps

            engine = BacktestEngine(
                starting_cash=starting_cash, slippage_bps=SLIPPAGE_BPS,
                dynamic_size_fn=lambda cash, f=fraction: f,
                deposits=deposits,
                variable_slippage_fn=tracked_slippage,
                regular_hours_only=REGULAR_HOURS_ONLY,
            )
            result = engine.run(bars, strategy)
            try:
                report = compute_report(result.equity_curve, result.trades, periods_per_year=ppy)
            except ValueError:
                print(f"{label:<32} not enough data")
                continue

            end_equity = result.equity_curve.iloc[-1]
            weekly = result.equity_curve.resample("7D").last().dropna()
            all_curves[frac_key][label] = [[ts.strftime("%Y-%m-%d"), round(float(v), 2)] for ts, v in weekly.items()]

            avg_slip = sum(slip_samples) / len(slip_samples) if slip_samples else 0.0
            max_slip = max(slip_samples) if slip_samples else 0.0
            print(f"{label:<32}{end_equity:>14,.2f}{report.num_trades:>8}{avg_slip:>14.2f}{max_slip:>14.2f}")
            comparison_rows.append({
                "fraction": frac_key, "combo": label, "end_equity_liquidity_aware": round(end_equity, 2),
                "avg_slip_bps": round(avg_slip, 3), "max_slip_bps": round(max_slip, 3),
            })
        print()

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump({
            "starting_cash": starting_cash, "deposit_dates": [d.isoformat() for d in deposit_dates],
            "monthly_deposit": MONTHLY_DEPOSIT, "from_date": from_date.isoformat(), "to_date": to_date.isoformat(),
            "fractions": [f"{f*100:.0f}%" for f in FRACTIONS], "curves": all_curves,
            "model": {"base_bps": BASE_BPS, "impact_bps_per_pct": IMPACT_BPS_PER_PCT, "cap_bps": CAP_BPS},
        }, f)
    print(f"Saved: {OUT_JSON}")

    # Direct comparison against the flat-2bps run already on disk, if present.
    flat_path = Path(__file__).resolve().parent / "alpaca_2year_curves.json"
    if flat_path.exists():
        flat = json.loads(flat_path.read_text(encoding="utf-8"))
        print("\n=== FLAT 2bps vs LIQUIDITY-AWARE, end balance $ ===")
        print(f"{'fraction':<10}{'combo':<32}{'flat $':>14}{'liquidity $':>14}{'difference':>14}")
        print("-" * 84)
        for row in comparison_rows:
            frac, combo = row["fraction"], row["combo"]
            flat_curve = flat.get("curves", {}).get(frac, {}).get(combo)
            if not flat_curve:
                continue
            flat_end = flat_curve[-1][1]
            liq_end = row["end_equity_liquidity_aware"]
            print(f"{frac:<10}{combo:<32}{flat_end:>14,.2f}{liq_end:>14,.2f}{liq_end - flat_end:>+14,.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
