"""What a stop actually BUYS you, separately from what it costs.

sweep_protective_exits.py answers "which config makes the most money" and says
stops make less. That is not the only question worth asking: a stop trades
expected return for a smaller worst case, and those are different objectives.
This reports both together — total P&L AND the tail — so the trade-off is an
explicit choice rather than an unexamined one.

Run:  python sweep_tail_risk.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from backtester import roster  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

WINDOWS = [("Jun", "2026-06-01", "2026-06-30"),
           ("Jul", "2026-07-01", "2026-07-31"),
           ("Aug", "2026-08-01", "2026-08-13")]
CONFIGS = [
    ("baseline (today)", {}),
    ("stop 1.0%", {"stop_loss_pct": 0.010}),
    ("stop 2.0%", {"stop_loss_pct": 0.020}),
    ("stop 3.0%", {"stop_loss_pct": 0.030}),
    ("tp 1.5%", {"take_profit_pct": 0.015}),
    ("stop 2.0% + tp 1.5%", {"stop_loss_pct": 0.020, "take_profit_pct": 0.015}),
]
DOLLARS_PER_TRADE = 1_000.0


class _Rec(Strategy):
    def __init__(self, inner):
        self.inner, self.signals = inner, []

    def on_bar(self, h, c):
        s = self.inner.on_bar(h, c)
        self.signals.append(s)
        return s

    def required_lookback(self):
        return self.inner.required_lookback()


class _Replay(Strategy):
    def __init__(self, sigs):
        self.signals, self.i = sigs, -1

    def on_bar(self, h, c):
        self.i += 1
        return self.signals[self.i] if self.i < len(self.signals) else Signal.HOLD

    def required_lookback(self):
        return Lookback(bars=1)


def _eng(**kw):
    return BacktestEngine(
        starting_cash=10_000.0, slippage_bps=2.0,
        fixed_dollars_per_trade=DOLLARS_PER_TRADE, **kw
    )


def main() -> int:
    combos = [(e.ticker, e.strategy_name, dict(e.params or {}))
              for e in roster.load_roster().entries if e.status == "active"]
    client = PolygonClient()

    cache = {}
    for label, start, end in WINDOWS:
        for ticker, strat, params in combos:
            try:
                bars = client.get_aggregates(ticker, start, end, 1, "minute")
            except Exception:  # noqa: BLE001
                continue
            if bars.empty or len(bars) < 100:
                continue
            rec = _Rec(build_strategy(strat, params))
            _eng().run(bars, rec)
            cache[(label, ticker)] = (bars, rec.signals)
    print(f"cached {len(cache)} (window, ticker) pairs\n")

    print(f"{'config':<22}{'window':>7}{'total':>10}{'/trade':>9}{'WORST':>10}{'5th pct':>10}{'>2% loss':>10}")
    print("-" * 78)
    for name, kw in CONFIGS:
        for label, _, _ in WINDOWS:
            pnls, rets = [], []
            for ticker, _, _ in combos:
                got = cache.get((label, ticker))
                if not got:
                    continue
                bars, sigs = got
                for t in _eng(**kw).run(bars, _Replay(list(sigs))).trades:
                    if t.pnl is None:
                        continue
                    pnls.append(t.pnl)
                    rets.append((t.exit_price - t.entry_price) / t.entry_price * 100 * (1 if t.shares > 0 else -1))
            if not pnls:
                continue
            worst = min(pnls)
            p5 = statistics.quantiles(pnls, n=20)[0] if len(pnls) > 20 else worst
            bad = sum(1 for r in rets if r < -2.0)
            print(f"{name:<22}{label:>7}{sum(pnls):>10.2f}{sum(pnls)/len(pnls):>9.3f}"
                  f"{worst:>10.2f}{p5:>10.2f}{bad:>10}")
        print()

    print("WORST = biggest single losing trade, in dollars on a $1,000 position.")
    print("5th pct = the trade at the 5th percentile (a routine bad day, not the freak one).")
    print("'>2% loss' = how many trades lost more than 2% of the position.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
