"""run_scan's opt-in regular_hours_only (added 2026-11-06): off by default (every existing scan unchanged), and when on it reaches the engine and stops new entries outside 09:30-16:00 New York.

    .venv/Scripts/python test_scanner_regular_hours.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import scanner  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402

PASS, FAIL = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -  {detail}" if detail and not ok else ""))


def bars(dip_at_utc: str, n_before: int = 120, n_total: int = 400) -> pd.DataFrame:
    """Perfectly flat price, one sharp dip of 3 bars starting at `dip_at_utc` (UTC, 2026-03-02: New York is UTC-5, so 14:30 UTC = the 09:30 open)."""
    idx = pd.date_range("2026-03-02 13:00", periods=n_total, freq="1min", tz="UTC")
    c = np.full(n_total, 100.0)
    k = idx.get_loc(pd.Timestamp(dip_at_utc, tz="UTC"))
    c[k:k + 3] = 97.0
    return pd.DataFrame({"open": c, "high": c + 0.02, "low": c - 0.02, "close": c, "volume": 100_000.0}, index=idx)


class Fake:
    def __init__(self, df):
        self.df = df

    def get_aggregates(self, ticker, from_date, to_date, multiplier=1, timespan="minute", **_):
        return self.df.copy()


def scan(df, **kw):
    return scanner.run_scan(["XYZ"], ["Bollinger Mean Reversion"], "2026-03-02", "2026-03-02", Fake(df), strategy_params={"Bollinger Mean Reversion": {"period": 20, "num_std": 2.0}}, max_workers=1, **kw)


def main() -> int:
    print("the setting")
    sig = inspect.signature(scanner.run_scan)
    check("it exists and is OFF by default", "regular_hours_only" in sig.parameters and sig.parameters["regular_hours_only"].default is False)

    print("it reaches the engine")
    seen = []

    class Spy(BacktestEngine):
        def __init__(self, *a, **k):
            seen.append(k.get("regular_hours_only", "absent"))
            super().__init__(*a, **k)

    orig = scanner.BacktestEngine
    scanner.BacktestEngine = Spy
    try:
        d = bars("2026-03-02 13:40")
        scan(d)
        scan(d, regular_hours_only=True)
        scan(d, regular_hours_only=False)
    finally:
        scanner.BacktestEngine = orig
    check("the engine is built exactly as before by default (no new keyword), with True when asked, and as before when asked for False", seen == ["absent", True, "absent"], str(seen))

    print("the effect")
    pre = bars("2026-03-02 13:40")                      # 08:40 New York: pre-market dip
    rth = bars("2026-03-02 15:30")                      # 10:30 New York: regular-hours dip
    r_pre_default, r_pre_flag = scan(pre), scan(pre, regular_hours_only=True)
    check("a pre-market dip is traded by default", r_pre_default[0].num_trades >= 1, str(r_pre_default[0].num_trades))
    check("and not at all with regular_hours_only (no new entry before 09:30)", r_pre_flag[0].num_trades == 0, str(r_pre_flag[0].num_trades))
    r_rth_default, r_rth_flag = scan(rth), scan(rth, regular_hours_only=True)
    check("a regular-hours dip is traded either way", r_rth_default[0].num_trades >= 1 and r_rth_flag[0].num_trades >= 1)
    check("and the result is identical either way when nothing happens outside regular hours", (r_rth_default[0].total_return, r_rth_default[0].num_trades, r_rth_default[0].sharpe_ratio) == (r_rth_flag[0].total_return, r_rth_flag[0].num_trades, r_rth_flag[0].sharpe_ratio))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
