"""Rolling walk-forward on protective exits — the real test, not one lucky split.

Run:  python sweep_rolling_walkforward.py [--start 2025-03] [--end 2026-08]

WHAT THIS ANSWERS THAT THE FIRST SWEEP DIDN'T
sweep_protective_exits.py did ONE select/test split (train June, test Jul+Aug)
and the selected config won 2/2. That is far too little to distinguish a real
effect from a coincidence. This walks a select/test pair across every
consecutive month pair available, so the question becomes "how OFTEN does the
training-window winner beat baseline out of sample" — a number that a single
lucky window cannot fake.

It also reports which VALUE got picked each time. That matters more than the
win count here: the first sweep chose 1.5% on June, 0.5% on July and 0.75% on
August. A parameter whose optimum moves that much every month is unstable, and
an unstable parameter that still wins on average is telling you the DIRECTION
is real while the NUMBER is not. If it stays unstable across a long run, that
is a finding — not a reason to keep hunting for a better value.

TWO METHOD CHOICES WORTH KNOWING
- The ticker set is PINNED here, not read from the live roster. The roster
  changes on its own (Q was paused mid-session on 2026-08-14), and a set that
  shifts underneath a validation run makes results incomparable between runs.
- Signals are computed once per (ticker, window) and replayed across the grid,
  verified identical to a normal run first. Same shortcut as the first sweep;
  it is what makes a grid this size affordable.
"""

from __future__ import annotations

import statistics
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

# Pinned deliberately — see module docstring. These are the combos that have
# actually been active or live-traded, with the params the roster runs them on.
COMBOS = [
    ("Q",    "VWAP Mean Reversion",      {"min_bars": 5, "entry_deviation_pct": 0.3}),
    ("PSKY", "VWAP Mean Reversion",      {"min_bars": 5, "entry_deviation_pct": 0.3}),
    ("SNDK", "VWAP Mean Reversion",      {"min_bars": 5, "entry_deviation_pct": 0.3}),
    ("BEN",  "Bollinger Mean Reversion", {"period": 15, "num_std": 3.0}),
    ("WTW",  "Bollinger Mean Reversion", {"period": 15, "num_std": 3.0}),
    ("SPCX", "Bollinger Mean Reversion", {"period": 15, "num_std": 3.0}),
]

STOPS = [None, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03]
TARGETS = [None, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03]
EOD = [False, True]
# TRAILING (2026-08-28) — added to test whether locking in a real recovery
# beats capping it (take_profit_pct) or letting it ride to the strategy's
# own signal (baseline). Bracketed around the ~0.5-0.75% region the ORIGINAL,
# now-refuted take_profit sweep referenced (see module docstring's "PROTECTIVE
# EXITS: FULLY REFUTED" note) — testing both directions rather than assuming
# it, the same discipline that refuted take_profit in the first place.
# (arm, trail) pairs, not a full cross product: trail values are only
# meaningful once arm is set, so (None, None) is the single "off" state.
TRAILING_STATES: list[tuple[float | None, float | None]] = [(None, None)] + [
    (arm, trail)
    for arm in (0.0025, 0.005, 0.0075)
    for trail in (0.0025, 0.005, 0.0075, 0.01)
]
BASELINE = (None, None, False, None, None)

STARTING_CASH = 10_000.0
DOLLARS_PER_TRADE = 1_000.0
SLIPPAGE_BPS = 2.0


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
        starting_cash=STARTING_CASH, slippage_bps=SLIPPAGE_BPS,
        fixed_dollars_per_trade=DOLLARS_PER_TRADE, **kw,
    )


def months(start: str, end: str) -> list[tuple[str, str, str]]:
    """[(label, first_day, last_day)] inclusive of both ends."""
    y, m = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    while (y, m) <= (ey, em):
        first = date(y, m, 1)
        nxt = date(y + (m == 12), 1 if m == 12 else m + 1, 1)
        out.append((f"{y}-{m:02d}", first.isoformat(), (nxt - timedelta(days=1)).isoformat()))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def main(argv: list[str]) -> int:
    start, end = "2025-03", "2026-08"
    tickers: set[str] | None = None
    for i, a in enumerate(argv):
        if a == "--start" and i + 1 < len(argv):
            start = argv[i + 1]
        if a == "--end" and i + 1 < len(argv):
            end = argv[i + 1]
        if a == "--tickers" and i + 1 < len(argv):
            tickers = {t.strip().upper() for t in argv[i + 1].split(",") if t.strip()}

    # Restricting the ticker set does NOT change COMBOS (still pinned — see
    # module docstring); it only filters which of the pinned combos run, so a
    # comparability run (e.g. --tickers BEN,WTW, the only two with full
    # coverage of every window) can't silently pick up a different param set.
    combos = [c for c in COMBOS if tickers is None or c[0] in tickers] if tickers else COMBOS
    unknown = (tickers or set()) - {c[0] for c in COMBOS}
    if unknown:
        print(f"Unknown ticker(s) not in the pinned COMBOS list: {sorted(unknown)}")
        return 2

    windows = months(start, end)
    # Trailing is NOT crossed with the full stop x target x eod grid — a
    # smoke test crossing all of it (8x8x2x13 = 1664 configs) took ~33
    # MINUTES for a single (window, 2-ticker) pair; a full 24-window/6-combo
    # run at that size would take the better part of two days, not "this
    # morning." Since replay cost is per-bar Python, not vectorized, cutting
    # the grid is the only lever available without rewriting the engine's
    # inner loop (out of scope for this pass).
    #
    # Instead: the existing 128-config stop x target x eod grid (unchanged,
    # already proven to run in reasonable time) PLUS trailing tested against
    # baseline directly (stop/target/eod all off) and against a SMALL,
    # representative stop_loss floor set — not the full 8-value STOPS grid —
    # since Stage 4's live design already decided trailing requires a
    # broker-resting stop_loss_pct floor anyway, so "trailing + every
    # possible stop" isn't the comparison that actually matters; "does
    # trailing help at all, under a stop floor close to what would ship" is.
    grid = [(s, t, e, None, None) for s in STOPS for t in TARGETS for e in EOD]
    grid += [
        (stop, None, False, arm, trail)
        for stop in (None, 0.01, 0.02)
        for (arm, trail) in TRAILING_STATES
        if arm is not None  # (None, None) is already in the base grid via (s, t, e, None, None)
    ]
    grid = list(dict.fromkeys(grid))  # de-dupe (e.g. stop=None+trailing-off already in the base grid)
    print(f"{len(windows)} monthly windows {start}..{end}, {len(combos)} combos "
          f"({', '.join(c[0] for c in combos)}), {len(grid)} configs\n")

    client = PolygonClient(use_cache=True)
    cache: dict[tuple, tuple] = {}
    t0 = time.time()
    for label, first, last in windows:
        for ticker, strat, params in combos:
            try:
                bars = client.get_aggregates(ticker, first, last, 1, "minute")
            except Exception as e:  # noqa: BLE001
                print(f"  {ticker:<6} {label}  skipped ({str(e)[:50]})", flush=True)
                continue
            if bars.empty or len(bars) < 500:
                print(f"  {ticker:<6} {label}  skipped (only {len(bars)} bars)", flush=True)
                continue
            rec = _Rec(build_strategy(strat, params))
            base = _eng().run(bars, rec)
            if not _eng().run(bars, _Replay(list(rec.signals))).equity_curve.equals(base.equity_curve):
                print(f"  {ticker:<6} {label}  REPLAY MISMATCH — skipped", flush=True)
                continue
            cache[(label, ticker)] = (bars, rec.signals)
        print(f"  fetched {label}  ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\ncached {len(cache)} (window, ticker) pairs in {time.time()-t0:.0f}s")
    print("running grid...\n", flush=True)

    # results[window][config] = total pnl
    results: dict[str, dict[tuple, float]] = {}
    for label, _, _ in windows:
        if not any(k[0] == label for k in cache):
            continue
        results[label] = {}
        for cfg in grid:
            stop, target, eod, arm, trail = cfg
            total = 0.0
            for ticker, _, _ in combos:
                got = cache.get((label, ticker))
                if not got:
                    continue
                bars, sigs = got
                r = _eng(stop_loss_pct=stop, take_profit_pct=target, close_at_session_end=eod,
                         trailing_arm_pct=arm, trailing_stop_pct=trail).run(bars, _Replay(list(sigs)))
                total += sum(t.pnl for t in r.trades if t.pnl is not None)
            results[label][cfg] = total
        print(f"  {label} done ({time.time()-t0:.0f}s)", flush=True)

    labels = [w[0] for w in windows if w[0] in results]
    if len(labels) < 2:
        print("\nNot enough windows with data to walk forward.")
        return 1

    def fmt(cfg):
        s, t, e, a, tr = cfg
        return (f"stop={'-' if s is None else f'{s*100:g}%'}/"
                f"tp={'-' if t is None else f'{t*100:g}%'}/"
                f"eod={'Y' if e else 'n'}/"
                f"trail={'-' if a is None else f'{a*100:g}%>{tr*100:g}%'}")

    print("\n" + "=" * 96)
    print("ROLLING WALK-FORWARD — select on each month, trade the next")
    print("=" * 96)
    print(f"  {'train':<9}{'test':<9}{'selected on train':<28}{'test $':>10}{'baseline $':>12}   verdict")
    wins, deltas, picks = 0, [], []
    for train, test in zip(labels, labels[1:]):
        best = max(results[train], key=lambda c: results[train][c])
        sel, base = results[test][best], results[test][BASELINE]
        better = sel > base
        wins += better
        deltas.append(sel - base)
        picks.append(best)
        print(f"  {train:<9}{test:<9}{fmt(best):<28}{sel:>10.2f}{base:>12.2f}   "
              f"{'BETTER' if better else 'worse '} {sel-base:+.2f}")

    n = len(deltas)
    print(f"\n  beat baseline out-of-sample: {wins}/{n} windows ({wins/n*100:.0f}%)")
    print(f"  mean edge per window: ${statistics.mean(deltas):+.2f}"
          + (f"   median ${statistics.median(deltas):+.2f}" if n > 2 else ""))
    if n > 2:
        print(f"  std dev: ${statistics.pstdev(deltas):.2f}  "
              f"(bigger than the mean = noise, not edge)")

    print("\n" + "=" * 96)
    print("PARAMETER STABILITY — how much did the winner move month to month?")
    print("=" * 96)
    for kind, idx in (("stop", 0), ("take-profit", 1)):
        vals = [p[idx] for p in picks]
        chosen = [f"{v*100:g}%" if v is not None else "off" for v in vals]
        distinct = len(set(chosen))
        # Distinct-value COUNT alone is a bad stability signal — 8 distinct
        # values out of 23 windows sounds bad but says nothing about whether
        # one value actually dominates. Mode SHARE (how often the single most
        # common pick was chosen) is what "stable" should mean: a bare
        # majority landing on one value is a real signal, and the >50% bar
        # here is deliberately the same threshold used for baseline "better/
        # worse" counts elsewhere in this report, not picked to flatter the
        # result.
        mode_value, mode_count = statistics.mode(chosen), chosen.count(statistics.mode(chosen))
        mode_share = mode_count / len(chosen)
        print(f"  {kind:<12} picked: {', '.join(chosen)}")
        print(f"  {'':<12} {distinct} distinct value(s); most common '{mode_value}' "
              f"in {mode_count}/{len(chosen)} windows ({mode_share*100:.0f}%)"
              + ("  <- stable" if mode_share > 0.5 else "  <- UNSTABLE; the number moves too "
                 "much to trust, even if the direction has an edge"))
    # Trailing gets its OWN stability pass, same rigor, since it's a genuinely
    # separate mechanism from stop/take-profit above, not a variant of either.
    for kind, idx in (("trail-arm", 3), ("trail-stop", 4)):
        vals = [p[idx] for p in picks]
        chosen = [f"{v*100:g}%" if v is not None else "off" for v in vals]
        distinct = len(set(chosen))
        # Distinct-value COUNT alone is a bad stability signal — 8 distinct
        # values out of 23 windows sounds bad but says nothing about whether
        # one value actually dominates. Mode SHARE (how often the single most
        # common pick was chosen) is what "stable" should mean: a bare
        # majority landing on one value is a real signal, and the >50% bar
        # here is deliberately the same threshold used for baseline "better/
        # worse" counts elsewhere in this report, not picked to flatter the
        # result.
        mode_value, mode_count = statistics.mode(chosen), chosen.count(statistics.mode(chosen))
        mode_share = mode_count / len(chosen)
        print(f"  {kind:<12} picked: {', '.join(chosen)}")
        print(f"  {'':<12} {distinct} distinct value(s); most common '{mode_value}' "
              f"in {mode_count}/{len(chosen)} windows ({mode_share*100:.0f}%)"
              + ("  <- stable" if mode_share > 0.5 else "  <- UNSTABLE; the number moves too "
                 "much to trust, even if the direction has an edge"))
    eod_picks = sum(1 for p in picks if p[2])
    print(f"  session-end  chosen in {eod_picks}/{len(picks)} windows")

    print("\n" + "=" * 96)
    print("EACH SETTING ON ITS OWN, AVERAGED OVER EVERY WINDOW (no selection step)")
    print("=" * 96)
    print("  Selection-free, so it can't be fooled by picking a winner after the fact.")
    base_mean = statistics.mean(results[l][BASELINE] for l in labels)
    print(f"  {'config':<26}{'mean $/window':>15}{'vs baseline':>14}{'windows better':>17}")
    for cfg in [BASELINE] + [c for c in grid if c != BASELINE]:
        vals = [results[l][cfg] for l in labels]
        mean = statistics.mean(vals)
        better = sum(1 for l in labels if results[l][cfg] > results[l][BASELINE])
        if cfg != BASELINE and abs(mean - base_mean) < 0.01:
            continue
        tag = "  <- baseline" if cfg == BASELINE else ""
        print(f"  {fmt(cfg):<26}{mean:>15.2f}{mean-base_mean:>+14.2f}"
              f"{better:>12}/{len(labels)}{tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
