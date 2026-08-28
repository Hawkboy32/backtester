"""Choose stop-loss / take-profit / session-end settings from data, not guesswork.

Run:  python sweep_protective_exits.py

METHOD
Signals are a pure function of price history (no position state), so each
(ticker, strategy, window) has its signals computed ONCE and then replayed
across the whole parameter grid — verified identical to a normal run before
being trusted, and ~18x faster, which is what makes a real grid affordable.

Parameters are SELECTED on the training window and REPORTED on later,
non-overlapping windows. Selecting and reporting on the same window is how a
sweep talks itself into a number that never repeats live — this session
already had one walk-forward result reverse itself when the window moved, so
the out-of-sample column is the only one worth believing.

Sizing is a fixed dollar amount per trade rather than the engine's default
all-in compounding, so totals are additive across combos and a config can't
win just by compounding a lucky early trade.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from backtester import roster  # noqa: E402
from backtester.auto_trader_state import load_control  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.engine import BacktestEngine  # noqa: E402
from backtester.strategies import build_strategy  # noqa: E402
from backtester.strategy import Lookback, Signal, Strategy  # noqa: E402

WINDOWS = [
    ("TRAIN  Jun", "2026-06-01", "2026-06-30"),
    ("TEST   Jul", "2026-07-01", "2026-07-31"),
    ("TEST   Aug", "2026-08-01", "2026-08-13"),
    # Cross-checked against the REAL live_trades.db ledger below (see
    # _print_sanity_check) - added 2026-08-27 specifically to catch a
    # replay-fidelity problem before trusting the grid's conclusions on it.
    ("SANITY Aug13-27", "2026-08-13", "2026-08-27"),
]
STOPS = [None, 0.005, 0.0075, 0.01, 0.015, 0.02]
TARGETS = [None, 0.005, 0.0075, 0.01, 0.015]
EOD = [False, True]

STARTING_CASH = 10_000.0
DOLLARS_PER_TRADE = 1_000.0
SLIPPAGE_BPS = 2.0


class _Recorder(Strategy):
    def __init__(self, inner):
        self.inner, self.signals = inner, []

    def on_bar(self, history, current):
        s = self.inner.on_bar(history, current)
        self.signals.append(s)
        return s

    def required_lookback(self):
        return self.inner.required_lookback()


class _Replay(Strategy):
    """Replays recorded signals. Declares a 1-bar lookback because it genuinely
    needs no history — that is what removes the O(n) slice per bar."""

    def __init__(self, signals):
        self.signals, self.i = signals, -1

    def on_bar(self, history, current):
        self.i += 1
        return self.signals[self.i] if self.i < len(self.signals) else Signal.HOLD

    def required_lookback(self):
        return Lookback(bars=1)


def _engine(**kw) -> BacktestEngine:
    return BacktestEngine(
        starting_cash=STARTING_CASH,
        slippage_bps=SLIPPAGE_BPS,
        fixed_dollars_per_trade=DOLLARS_PER_TRADE,
        **kw,
    )


def _live_combos() -> list[tuple[str, str, dict]]:
    """Every (ticker, strategy_name, params) this account actually trades
    right now: the active roster (equities) PLUS every extra_targets group
    (crypto/forex, tuned separately from the roster - see control.json).
    Reads the real live config rather than a hand-maintained ticker list, so
    the sweep can't silently drift from what auto_trader.py is doing.

    KNOWN LIMITATION, tried and reverted 2026-08-27: this freezes TODAY's
    active roster across every backtest window, including windows before
    some of today's tickers were ever promoted, or spanning stretches where
    a ticker was actually paused live (Q and MPWR both were, repeatedly,
    during Aug13-27). A reconstruction from roster_events.jsonl was built to
    fix this and then abandoned: cross-checked against roster.json's own
    promoted_at field, PSKY's real re-promotion at 2026-08-13T23:10:56 shows
    up in NEITHER the event log NOR a reliable promoted_at (MPWR's is even
    None despite being active right now) - the data needed to reconstruct
    exact historical roster state doesn't reliably exist yet. A wrong
    reconstruction would have silently zeroed out real active periods,
    making the sanity-check gap worse, not better - so this stays a known,
    documented gap rather than a subtly broken fix. Fixing it properly means
    fixing roster.py's own state logging first, not patching around it here.
    """
    state = roster.load_roster()
    combos = [(e.ticker, e.strategy_name, dict(e.params or {}))
              for e in state.entries if e.status == "active"]
    control = load_control()
    for group in control.extra_targets:
        strategy_name = group.get("strategy_name", "")
        params = dict(group.get("strategy_params", {}))
        for ticker in group.get("tickers", []):
            combos.append((ticker, strategy_name, params))
    return combos


def main() -> int:
    combos = _live_combos()
    if not combos:
        print("No active roster combos.")
        return 1

    print(f"{len(combos)} active combos: " + ", ".join(f"{t}/{s.split()[0]}" for t, s, _ in combos))
    print(f"grid: {len(STOPS)} stops x {len(TARGETS)} targets x {len(EOD)} eod = "
          f"{len(STOPS)*len(TARGETS)*len(EOD)} configs, over {len(WINDOWS)} windows\n")

    client = PolygonClient()
    cache: dict[tuple, tuple] = {}

    # ---- record signals once per (combo, window), verifying the shortcut ----
    for label, start, end in WINDOWS:
        for ticker, strat_name, params in combos:
            try:
                bars = client.get_aggregates(ticker, start, end, 1, "minute")
            except Exception as e:  # noqa: BLE001
                print(f"  {ticker} {label}: no data ({e})")
                continue
            if bars.empty or len(bars) < 100:
                print(f"  {ticker} {label}: too few bars ({len(bars)})")
                continue
            rec = _Recorder(build_strategy(strat_name, params))
            t0 = time.time()
            base = _engine().run(bars, rec)
            replay = _engine().run(bars, _Replay(list(rec.signals)))
            if not base.equity_curve.equals(replay.equity_curve):
                print(f"  {ticker} {label}: REPLAY MISMATCH — signals are position-dependent, "
                      "cannot use the cached-signal shortcut for this strategy. Skipped.")
                continue
            cache[(label, ticker, strat_name)] = (bars, rec.signals)
            print(f"  recorded {ticker:<6} {label}  {len(bars):>6} bars  "
                  f"{len(base.trades):>4} trades  ({time.time()-t0:.1f}s)")

    max_trades_per_day = load_control().max_trades_per_day
    print(f"enforcing the real live cap: max_trades_per_day={max_trades_per_day} "
          f"(global, across all combos combined - a fix 2026-08-27, the original sweep "
          f"had no cap and ran ~17/day where live is capped at {max_trades_per_day})")

    print("\nrunning grid...")
    # results[(stop, target, eod)][window_label] = (total_pnl, n_trades, reason_counts)
    results: dict[tuple, dict[str, list]] = {}
    t_start = time.time()
    for stop in STOPS:
        for target in TARGETS:
            for eod in EOD:
                key = (stop, target, eod)
                results[key] = {}
                for label, _, _ in WINDOWS:
                    # Collect every closed trade across all combos for this
                    # (config, window) FIRST, with its entry_time, so the
                    # daily cap can be enforced chronologically across the
                    # whole account - not per-ticker, matching how
                    # auto_trader.py's status.trades_today actually counts.
                    all_trades = []
                    for ticker, strat_name, _ in combos:
                        entry = cache.get((label, ticker, strat_name))
                        if entry is None:
                            continue
                        bars, signals = entry
                        r = _engine(
                            stop_loss_pct=stop, take_profit_pct=target, close_at_session_end=eod
                        ).run(bars, _Replay(list(signals)))
                        all_trades.extend(t for t in r.trades if t.pnl is not None)
                    all_trades.sort(key=lambda t: t.entry_time)

                    total, n, reasons = 0.0, 0, {}
                    day_counts: dict = {}
                    for t in all_trades:
                        day = t.entry_time.date()
                        count_today = day_counts.get(day, 0)
                        if count_today >= max_trades_per_day:
                            continue  # cap reached for this day - live wouldn't have taken it either
                        day_counts[day] = count_today + 1
                        total += t.pnl
                        n += 1
                        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1
                    results[key][label] = [total, n, reasons]
    print(f"grid done in {time.time()-t_start:.0f}s\n")

    train_label = WINDOWS[0][0]
    test_labels = [w[0] for w in WINDOWS[1:]]
    baseline = (None, None, False)

    def fmt(k):
        s, tp, e = k
        return f"stop={'-' if s is None else f'{s*100:.2f}%':>6}  tp={'-' if tp is None else f'{tp*100:.2f}%':>6}  eod={'Y' if e else 'n'}"

    print("=" * 92)
    print("BASELINE (no protective exits) — what the bot does today")
    print("=" * 92)
    for label, _, _ in WINDOWS:
        tot, n, _ = results[baseline][label]
        print(f"  {label}:  total ${tot:>9.2f}  over {n:>4} trades  "
              f"= ${tot/n if n else 0:>7.3f}/trade")

    print("\n" + "=" * 92)
    print(f"TOP 10 CONFIGS BY {train_label} (selection window) — with their OUT-OF-SAMPLE results")
    print("=" * 92)
    ranked = sorted(results, key=lambda k: results[k][train_label][0], reverse=True)[:10]
    header = f"  {'config':<40}" + "".join(f"{lbl:>16}" for lbl, _, _ in WINDOWS)
    print(header)
    for k in ranked:
        row = f"  {fmt(k):<40}"
        for label, _, _ in WINDOWS:
            tot, n, _ = results[k][label]
            row += f"{'$'+format(tot,'.2f'):>16}"
        print(row)

    print("\n" + "=" * 92)
    print("DOES THE TRAIN-SELECTED CONFIG BEAT THE BASELINE OUT OF SAMPLE?")
    print("=" * 92)
    best = ranked[0]
    print(f"  selected on {train_label}: {fmt(best)}")
    verdicts = []
    for label in test_labels:
        b_tot, b_n, _ = results[baseline][label]
        s_tot, s_n, reasons = results[best][label]
        better = s_tot > b_tot
        verdicts.append(better)
        print(f"\n  {label}")
        print(f"    baseline  ${b_tot:>9.2f}  ({b_n} trades)")
        print(f"    selected  ${s_tot:>9.2f}  ({s_n} trades)   -> {'BETTER' if better else 'WORSE'} "
              f"by ${abs(s_tot-b_tot):.2f}")
        print(f"    exits: {reasons}")
    print(f"\n  out-of-sample windows improved: {sum(verdicts)}/{len(verdicts)}")
    if sum(verdicts) < len(verdicts):
        print("  -> NOT consistent. Treat the training-window gain as fitting, not edge.")

    # Session-end on its own, holding everything else off — the one change with
    # a prior reason to expect an effect (carried positions averaged -2.5% live).
    print("\n" + "=" * 92)
    print("SESSION-END FLATTEN IN ISOLATION (no stop, no target)")
    print("=" * 92)
    for label, _, _ in WINDOWS:
        off = results[(None, None, False)][label]
        on = results[(None, None, True)][label]
        print(f"  {label}:  carry ${off[0]:>9.2f} ({off[1]:>4} trades)   "
              f"flatten ${on[0]:>9.2f} ({on[1]:>4} trades)   "
              f"diff ${on[0]-off[0]:>+9.2f}")

    _print_sanity_check(combos, results, baseline)
    return 0


def _print_sanity_check(combos, results, baseline) -> None:
    """Cross-check the BASELINE backtest replay's P&L for the sanity window
    against the REAL live_trades.db ledger for the same combos/dates. Not a
    penny-for-penny match test — real fills carry real slippage/spread the
    2bps model only approximates, and live position sizing isn't the fixed
    $1000/trade this sweep uses — but if the SIGN or rough scale disagrees,
    that's a replay-fidelity problem worth fixing before trusting anything
    else in this report, not something to wave past.
    """
    import sqlite3

    from backtester.live_trades import DB_PATH

    sanity = next((w for w in WINDOWS if w[0].startswith("SANITY")), None)
    if sanity is None:
        return
    label, start, end = sanity
    if label not in results.get(baseline, {}):
        return
    bt_total, bt_n, _ = results[baseline][label]

    con = sqlite3.connect(DB_PATH)
    real_total, real_n = 0.0, 0
    for ticker, strategy_name, _ in combos:
        row = con.execute(
            """SELECT COALESCE(SUM(pnl), 0), COUNT(*) FROM live_trades
               WHERE ticker = ? AND strategy_name = ? AND is_paper = 0
                 AND exit_time >= ? AND exit_time < ?""",
            (ticker, strategy_name, start, end + "T23:59:59"),
        ).fetchone()
        real_total += row[0]
        real_n += row[1]

    print("\n" + "=" * 92)
    print(f"SANITY CHECK — {label}: backtest baseline vs the REAL live_trades.db ledger")
    print("=" * 92)
    print(f"  backtest baseline (fixed $1000/trade, 2bps slippage): ${bt_total:>9.2f}  ({bt_n} trades)")
    print(f"  real ledger (is_paper=0, actual fills/sizing):        ${real_total:>9.2f}  ({real_n} decisions)")
    agree = (bt_total >= 0) == (real_total >= 0)
    print(f"  sign agreement: {'YES' if agree else 'NO — investigate before trusting the grid above'}")
    if real_n == 0:
        print("  (no real trades recorded for these combos/dates — nothing to cross-check against yet)")


if __name__ == "__main__":
    sys.exit(main())
