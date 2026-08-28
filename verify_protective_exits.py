"""Verification for the protective exits added to BacktestEngine 2026-08-14.

Run:  python verify_protective_exits.py

Covers the cases where a stop/target backtest most easily lies to itself:
intrabar triggering, a bar containing BOTH levels, gapping through a level,
and same-bar re-entry after a stop. Also proves the defaults are INERT, which
matters because every previously recorded scan result was produced by the
pre-change engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.engine import BacktestEngine, PositionMode  # noqa: E402
from backtester.strategy import Bar, Signal, Strategy  # noqa: E402


class ScriptedStrategy(Strategy):
    """Emits a pre-set signal per bar index, so a test controls exactly when
    entries and exits are attempted and the engine's own logic is what's
    under test rather than some indicator's."""

    def __init__(self, signals: list[Signal]):
        self.signals = signals
        self.i = -1

    def on_bar(self, history: pd.DataFrame, current: Bar) -> Signal:
        self.i += 1
        return self.signals[self.i] if self.i < len(self.signals) else Signal.HOLD


def make_bars(rows: list[tuple], start: str = "2026-01-05 09:30", freq: str = "1min") -> pd.DataFrame:
    """rows = [(open, high, low, close), ...]"""
    idx = pd.date_range(start=start, periods=len(rows), freq=freq, tz="UTC")
    return pd.DataFrame(
        [{"open": o, "high": h, "low": lo, "close": c, "volume": 1000} for o, h, lo, c in rows],
        index=idx,
    )


PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    # ---------------------------------------------------------------- inert
    print("\nDEFAULTS ARE INERT (no behaviour change for existing callers)")
    bars = make_bars([(100, 101, 99, 100), (100, 106, 94, 105), (105, 106, 99, 100), (100, 101, 99, 101)])
    sig = [Signal.BUY, Signal.HOLD, Signal.HOLD, Signal.SELL]

    baseline = BacktestEngine(starting_cash=10_000).run(bars, ScriptedStrategy(list(sig)))
    # Simulate the PRE-change engine by forcing the new hook to never fire.
    old = BacktestEngine(starting_cash=10_000)
    old._protective_exit = lambda *a, **k: None  # type: ignore[method-assign]
    old_result = old.run(bars, ScriptedStrategy(list(sig)))
    check(
        "default config == engine with protective exits disabled entirely",
        baseline.equity_curve.equals(old_result.equity_curve)
        and [t.pnl for t in baseline.trades] == [t.pnl for t in old_result.trades],
        f"{len(baseline.trades)} trades, final equity {baseline.equity_curve.iloc[-1]:.4f}",
    )

    # ------------------------------------------------------------ stop loss
    print("\nSTOP LOSS")
    # Entry at bar 0 close=100. Bar 1 dips to 94 (-6%) but CLOSES back at 105.
    # A close-only check would see +5% and miss the stop entirely.
    r = BacktestEngine(starting_cash=10_000, stop_loss_pct=0.05).run(bars, ScriptedStrategy(list(sig)))
    t = r.trades[0]
    check(
        "stop triggers INTRABAR on the low, not the close",
        t.exit_reason == "stop_loss" and abs(t.exit_price - 95.0) < 1e-9,
        f"exit {t.exit_price:.4f} reason={t.exit_reason} (bar closed at 105)",
    )

    # Gap: bar 1 OPENS at 90, below the 95 stop -> must fill at 90, not 95.
    gap_bars = make_bars([(100, 101, 99, 100), (90, 92, 88, 91)])
    r = BacktestEngine(starting_cash=10_000, stop_loss_pct=0.05).run(
        gap_bars, ScriptedStrategy([Signal.BUY, Signal.HOLD])
    )
    t = r.trades[0]
    check(
        "gapping THROUGH the stop fills at the open (worse), not at the stop",
        abs(t.exit_price - 90.0) < 1e-9,
        f"exit {t.exit_price:.4f} — filling at 95 would hide the gap loss",
    )

    # --------------------------------------------------------- take profit
    print("\nTAKE PROFIT")
    tp_bars = make_bars([(100, 101, 99, 100), (101, 108, 100, 102)])
    r = BacktestEngine(starting_cash=10_000, take_profit_pct=0.05).run(
        tp_bars, ScriptedStrategy([Signal.BUY, Signal.HOLD])
    )
    t = r.trades[0]
    check(
        "target triggers intrabar on the high",
        t.exit_reason == "take_profit" and abs(t.exit_price - 105.0) < 1e-9,
        f"exit {t.exit_price:.4f} reason={t.exit_reason}",
    )

    # ------------------------------------------ both levels in the same bar
    print("\nAMBIGUOUS BAR (contains both stop and target)")
    both = make_bars([(100, 101, 99, 100), (100, 108, 92, 104)])
    r = BacktestEngine(starting_cash=10_000, stop_loss_pct=0.05, take_profit_pct=0.05).run(
        both, ScriptedStrategy([Signal.BUY, Signal.HOLD])
    )
    t = r.trades[0]
    check(
        "resolves to the STOP (conservative) when OHLC can't say which came first",
        t.exit_reason == "stop_loss",
        f"reason={t.exit_reason} — assuming take_profit here is how stop/target backtests inflate",
    )

    # ------------------------------------------------------------ max hold
    print("\nMAX HOLD")
    flat = make_bars([(100, 100.5, 99.5, 100)] * 6)
    r = BacktestEngine(starting_cash=10_000, max_hold_bars=3).run(
        flat, ScriptedStrategy([Signal.BUY] + [Signal.HOLD] * 5)
    )
    t = r.trades[0]
    held = flat.index.get_loc(t.exit_time) - flat.index.get_loc(t.entry_time)
    check("exits after exactly max_hold_bars", t.exit_reason == "max_hold" and held == 3, f"held {held} bars")

    # --------------------------------------------------------- session end
    print("\nSESSION END")
    day1 = pd.date_range("2026-01-05 14:30", periods=3, freq="1min", tz="UTC")
    day2 = pd.date_range("2026-01-06 14:30", periods=3, freq="1min", tz="UTC")
    sess = pd.DataFrame(
        [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 10}] * 6,
        index=day1.append(day2),
    )
    r = BacktestEngine(starting_cash=10_000, close_at_session_end=True).run(
        sess, ScriptedStrategy([Signal.BUY] + [Signal.HOLD] * 5)
    )
    t = r.trades[0]
    check(
        "flattens on the last bar of the session, not carried overnight",
        t.exit_reason == "session_end" and t.exit_time == day1[-1],
        f"exit {t.exit_time} reason={t.exit_reason}",
    )
    r_off = BacktestEngine(starting_cash=10_000).run(
        sess, ScriptedStrategy([Signal.BUY] + [Signal.HOLD] * 5)
    )
    check(
        "...and WITHOUT the flag the same position is still carried (flag is what changed it)",
        r_off.trades[0].exit_time == day2[-1],
        f"exit {r_off.trades[0].exit_time}",
    )

    # ------------------------------------------------- no same-bar re-entry
    print("\nNO SAME-BAR RE-ENTRY AFTER A STOP")
    rb = make_bars([(100, 101, 99, 100), (100, 101, 90, 100), (100, 101, 99, 100)])
    r = BacktestEngine(starting_cash=10_000, stop_loss_pct=0.05).run(
        rb, ScriptedStrategy([Signal.BUY, Signal.BUY, Signal.HOLD])
    )
    check(
        "a BUY on the stop-out bar does NOT reopen that same bar",
        len(r.trades) == 1 and r.trades[0].exit_reason == "stop_loss",
        f"{len(r.trades)} trade(s) — >1 would mean a stop sweep invents round trips",
    )

    # ------------------------------------------------------------- shorts
    print("\nSHORT SIDE (levels mirror)")
    sh = make_bars([(100, 101, 99, 100), (100, 106, 99, 101)])
    r = BacktestEngine(
        starting_cash=10_000, stop_loss_pct=0.05, position_mode=PositionMode.SHORT_ONLY
    ).run(sh, ScriptedStrategy([Signal.SELL, Signal.HOLD]))
    t = r.trades[0]
    check(
        "short stop is ABOVE entry and triggers on the high",
        t.exit_reason == "stop_loss" and abs(t.exit_price - 105.0) < 1e-9,
        f"exit {t.exit_price:.4f}",
    )
    check("short stop loses money (sign check)", t.pnl < 0, f"pnl {t.pnl:.4f}")

    sh2 = make_bars([(100, 101, 99, 100), (100, 101, 94, 96)])
    r = BacktestEngine(
        starting_cash=10_000, take_profit_pct=0.05, position_mode=PositionMode.SHORT_ONLY
    ).run(sh2, ScriptedStrategy([Signal.SELL, Signal.HOLD]))
    t = r.trades[0]
    check(
        "short target is BELOW entry and makes money",
        t.exit_reason == "take_profit" and t.pnl > 0,
        f"exit {t.exit_price:.4f} pnl {t.pnl:+.4f}",
    )

    # ------------------------------------------------------------ slippage
    print("\nSLIPPAGE STILL APPLIES TO PROTECTIVE EXITS")
    r = BacktestEngine(starting_cash=10_000, stop_loss_pct=0.05, slippage_bps=100).run(
        bars, ScriptedStrategy(list(sig))
    )
    t = r.trades[0]
    check(
        "stop exit fills BELOW the stop once slippage is applied",
        t.exit_price < 95.0,
        f"exit {t.exit_price:.4f} vs stop 95.00 (1% slippage)",
    )

    # ---------------------------------------------------------- trailing stop
    print("\nTRAILING STOP (2026-08-28)")

    # Entry 100. Bar1 arms (peak 104, +4% >= 3% arm) without triggering the
    # 2% trail (low stays above the trail price). Bar2 sets a new peak (105)
    # THEN retraces enough from THAT peak to trigger — proves peak tracking
    # uses each bar's own high before checking, not last bar's.
    tr = make_bars([(100, 101, 99, 100), (103, 104, 102.5, 103.5), (103, 105, 101, 101.5)])
    r = BacktestEngine(starting_cash=10_000, trailing_arm_pct=0.03, trailing_stop_pct=0.02).run(
        tr, ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.HOLD])
    )
    t = r.trades[0]
    check(
        "arms on the bar that reaches the arm threshold, triggers on retracement from the PEAK",
        t.exit_reason == "trailing_stop" and abs(t.exit_price - 102.9) < 1e-9,
        f"exit {t.exit_price:.4f} reason={t.exit_reason} (peak 105, trail = 105*0.98 = 102.90)",
    )
    check("a trailing exit that protected real profit still shows positive pnl", t.pnl > 0, f"pnl {t.pnl:+.4f}")

    # Same shape, mirrored for a short — peak tracks the LOW since entry, the
    # trail sits ABOVE it, and a rally (not a dip) triggers it.
    tr_sh = make_bars([(100, 101, 99, 100), (97, 97.5, 96, 96.5), (97, 98.5, 96, 98)])
    r = BacktestEngine(
        starting_cash=10_000, trailing_arm_pct=0.03, trailing_stop_pct=0.02, position_mode=PositionMode.SHORT_ONLY,
    ).run(tr_sh, ScriptedStrategy([Signal.SELL, Signal.HOLD, Signal.HOLD]))
    t = r.trades[0]
    check(
        "short side: trail sits ABOVE the trough and triggers on a rally, not a dip",
        t.exit_reason == "trailing_stop" and abs(t.exit_price - 97.92) < 1e-9,
        f"exit {t.exit_price:.4f} reason={t.exit_reason} (trough 96, trail = 96*1.02 = 97.92)",
    )
    check("short trailing exit that protected real profit still shows positive pnl", t.pnl > 0, f"pnl {t.pnl:+.4f}")

    # Price never moves 3% in its favor — must NOT fire even on a large
    # retracement, since it never armed at all. Confirms the arm gate holds
    # regardless of how big a subsequent drawdown is.
    never_arms = make_bars(
        [(100, 101, 99, 100), (100, 101, 99, 100.5), (100, 100, 90, 95), (95, 96, 94, 95.5)]
    )
    r = BacktestEngine(starting_cash=10_000, trailing_arm_pct=0.03, trailing_stop_pct=0.02).run(
        never_arms, ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.HOLD, Signal.SELL])
    )
    check(
        "never arms (price never reaches the arm threshold) -> never fires, even on a big drawdown",
        all(t.exit_reason != "trailing_stop" for t in r.trades),
        f"exit_reasons: {[t.exit_reason for t in r.trades]}",
    )

    # Bar1 arms without triggering (low stays above the trail). Bar2 GAPS
    # OPEN below the trail price -> must fill at that bar's open, not the
    # trail level, same conservatism as the existing stop-loss gap case.
    tr_gap = make_bars([(100, 101, 99, 100), (103, 105, 103, 104), (100, 101, 98, 99)])
    r = BacktestEngine(starting_cash=10_000, trailing_arm_pct=0.02, trailing_stop_pct=0.02).run(
        tr_gap, ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.HOLD])
    )
    t = r.trades[0]
    check(
        "gapping THROUGH the trail level fills at the open (worse), not at the trail price",
        t.exit_reason == "trailing_stop" and abs(t.exit_price - 100.0) < 1e-9,
        f"exit {t.exit_price:.4f} — trail was at 102.90 (105*0.98); filling there would hide the gap",
    )

    # A single bar whose high reaches take_profit's target AND whose low
    # retraces enough from that SAME bar's new peak to trigger a tight,
    # already-armed trail. Confirmed precedence: TRAILING resolves first —
    # its exit is closer to entry (less profit) than take-profit's, the
    # conservative reading when a bar could satisfy either.
    ambiguous = make_bars([(100, 101, 99, 100), (109, 110, 105, 108)])
    r = BacktestEngine(
        starting_cash=10_000, trailing_arm_pct=0.02, trailing_stop_pct=0.01, take_profit_pct=0.10,
    ).run(ambiguous, ScriptedStrategy([Signal.BUY, Signal.HOLD]))
    t = r.trades[0]
    check(
        "a bar satisfying both trailing AND take_profit resolves to TRAILING (the smaller-profit outcome)",
        t.exit_reason == "trailing_stop" and abs(t.exit_price - 108.9) < 1e-9,
        f"reason={t.exit_reason} exit={t.exit_price:.4f} — take_profit would have given 110.00, "
        f"assuming that here is the self-flattering read this ordering exists to avoid",
    )

    # Same shape as the "no same-bar re-entry after a stop" case above, for
    # trailing specifically: a BUY signal on the exact bar the trail fires
    # must not reopen a position that same bar.
    tr_reentry = make_bars([(100, 101, 99, 100), (103, 104, 102.5, 103.5), (103, 105, 101, 101.5)])
    r = BacktestEngine(starting_cash=10_000, trailing_arm_pct=0.03, trailing_stop_pct=0.02).run(
        tr_reentry, ScriptedStrategy([Signal.BUY, Signal.HOLD, Signal.BUY])
    )
    check(
        "a BUY on the trailing-exit bar does NOT reopen that same bar",
        len(r.trades) == 1 and r.trades[0].exit_reason == "trailing_stop",
        f"{len(r.trades)} trade(s) — >1 would mean a trail sweep invents round trips",
    )

    # ------------------------------------------------------------ validation
    print("\nINPUT VALIDATION")
    for kwargs, label in [
        ({"stop_loss_pct": 0}, "stop_loss_pct=0"),
        ({"stop_loss_pct": -0.1}, "negative stop"),
        ({"max_hold_bars": 0}, "max_hold_bars=0"),
        ({"trailing_arm_pct": 0}, "trailing_arm_pct=0"),
        ({"trailing_stop_pct": 0.02}, "trailing_stop_pct set without trailing_arm_pct"),
    ]:
        try:
            BacktestEngine(**kwargs)
            check(f"rejects {label}", False, "no error raised")
        except ValueError:
            check(f"rejects {label}", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
