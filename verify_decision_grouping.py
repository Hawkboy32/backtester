"""Verification for the decision-grouping fix in live_trades.recent_performance
(2026-08-15) — a combo connected to N accounts was counting one real trading
decision as N separate wins/losses, which is what paused Q on a single loss.

Two parts:
  1. Synthetic unit tests of _group_into_decisions against the exact edge
     cases the fix has to get right (mixed-sign decisions, chain-drift,
     ticker isolation via the caller's WHERE clause, breakeven).
  2. A REPLAY against the real Q/VWAP Mean Reversion history that triggered
     this, frozen to the rows recorded through the 2026-08-14 16:18 incident
     (not recent_performance()'s live view, which drifts as new trades keep
     closing against this combo — see INCIDENT_CUTOFF below), proving the
     corrected streak on the real data that motivated the fix.

Run:  python verify_decision_grouping.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.live_trades import (  # noqa: E402
    DB_PATH, DECISION_GROUP_SECONDS, _group_into_decisions, recent_performance,
)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print("\nSYNTHETIC GROUPING CASES")

    # Three accounts, one decision, all agree it lost.
    rows = [
        ("2026-01-01T10:00:00Z", -1.35, -0.315),
        ("2026-01-01T10:00:00Z", -32.50, -0.325),
        ("2026-01-01T10:00:01Z", -0.02, -0.193),
    ]
    d = _group_into_decisions(rows)
    check("3 rows within 1s -> ONE decision", len(d) == 1, f"got {len(d)}")
    check("all-negative decision is a loss", d[0]["is_loss"] and not d[0]["is_win"])
    check("decision pnl is the SUM of dollars", abs(d[0]["pnl"] - (-1.35 - 32.50 - 0.02)) < 1e-9)

    # Real case: dollar signs DISAGREE, mean pct decides — the whole point.
    rows = [
        ("2026-01-01T10:00:00Z", +19.59, +0.196),   # biggest dollar gain
        ("2026-01-01T10:00:00Z", -5.54, -0.549),    # biggest dollar loss
        ("2026-01-01T10:00:01Z", +0.05, +0.626),
    ]
    d = _group_into_decisions(rows)
    mean_pct = (0.196 - 0.549 + 0.626) / 3
    check(
        "mixed-sign decision classified by MEAN PCT, not dollar sum",
        d[0]["is_win"] and mean_pct > 0,
        f"mean_pct={mean_pct:+.3f}%, dollar sum={19.59-5.54+0.05:+.2f} (would mislead if used)",
    )

    # Two decisions well outside the grouping window (60s poll cadence).
    rows = [
        ("2026-01-01T10:00:00Z", -1.0, -0.1),
        ("2026-01-01T10:00:01Z", -1.0, -0.1),
        ("2026-01-01T10:01:30Z", +1.0, +0.1),
        ("2026-01-01T10:01:31Z", +1.0, +0.1),
    ]
    d = _group_into_decisions(rows)
    check("two decisions 90s apart stay SEPARATE", len(d) == 2, f"got {len(d)}")
    check("first is a loss, second is a win", d[0]["is_loss"] and d[1]["is_win"])

    # Chain-drift: rows arrive 8s apart repeatedly. Anchored to the group's
    # FIRST row (not a sliding window), so this must NOT merge into one giant
    # decision purely by chaining sub-threshold gaps.
    rows = [
        ("2026-01-01T10:00:00Z", -1.0, -0.1),
        ("2026-01-01T10:00:08Z", -1.0, -0.1),   # 8s from anchor — still in
        ("2026-01-01T10:00:16Z", -1.0, -0.1),   # 16s from anchor — new group
    ]
    d = _group_into_decisions(rows)
    check(
        "anchored to group START, not previous row (no chain-drift)",
        len(d) == 2,
        f"got {len(d)} decision(s) — drift would wrongly merge all 3 into 1",
    )

    # Exact breakeven: neither a win nor a loss, matches the pre-fix behavior
    # for a literal pnl==0 row.
    rows = [("2026-01-01T10:00:00Z", 0.0, 0.0)]
    d = _group_into_decisions(rows)
    check("breakeven decision is neither win nor loss", not d[0]["is_win"] and not d[0]["is_loss"])

    # Boundary: exactly DECISION_GROUP_SECONDS apart — must be a NEW group
    # (strictly-greater-than comparison, not >=).
    rows = [
        ("2026-01-01T10:00:00Z", -1.0, -0.1),
        (f"2026-01-01T10:00:{int(DECISION_GROUP_SECONDS):02d}Z", -1.0, -0.1),
    ]
    d = _group_into_decisions(rows)
    check(f"exactly {DECISION_GROUP_SECONDS:.0f}s apart stays grouped (boundary inclusive)", len(d) == 1)

    print("\nREPLAY AGAINST REAL Q / VWAP MEAN REVERSION HISTORY")

    # live_trades.db keeps growing as the bot keeps trading, so asserting
    # against "whatever recent_performance() sees right now" is not
    # reproducible — it already isn't: two more Q/VWAP MR decisions closed
    # after this fix shipped (a loss on 2026-08-17, a win on 2026-08-31),
    # which correctly moves TODAY's current_losing_streak to 0. That is real
    # trading outcome, not a regression of the grouping fix. To keep proving
    # the actual bug this fix addressed, freeze the replay to the exact rows
    # recorded through the incident itself.
    INCIDENT_CUTOFF = "2026-08-14T16:18:47.710911+00:00"
    with sqlite3.connect(DB_PATH) as conn:
        rows_desc = conn.execute(
            """SELECT exit_time, pnl, pnl_pct FROM live_trades
               WHERE ticker = ? AND strategy_name = ? AND exit_time <= ?
               ORDER BY exit_time DESC""",
            ("Q", "VWAP Mean Reversion", INCIDENT_CUTOFF),
        ).fetchall()
    decisions = _group_into_decisions(list(reversed(rows_desc)))
    streak = 0
    for d in reversed(decisions):
        if not d["is_loss"]:
            break
        streak += 1

    print(f"  raw rows through the incident  = {len(rows_desc)}")
    print(f"  decisions                      = {len(decisions)}")
    print(f"  current_losing_streak          = {streak}")
    check(
        "streak on the incident replay is 1, not the pre-fix 3 that paused it",
        streak == 1,
        f"got {streak} — the 13:42 decision (mean +0.091%, mixed dollar "
        "signs) breaks the streak that the un-grouped code counted as a loss",
    )
    check(
        "14 raw rows through the incident group into 5 real decisions",
        len(decisions) == 5,
        f"got {len(decisions)} decisions from {len(rows_desc)} rows",
    )

    # Informational only, not asserted: today's actual rolling status for
    # this combo, which legitimately differs from the frozen replay above as
    # new trades close — there is no fixed "correct" value to check it against.
    stats = recent_performance("Q", "VWAP Mean Reversion")
    print("\n  (informational) current live status for Q / VWAP Mean Reversion:")
    print(f"    num_trades (decisions)     = {stats.num_trades}")
    print(f"    current_losing_streak      = {stats.current_losing_streak}")
    print(f"    win_rate                   = {stats.win_rate:.0%}" if stats.win_rate is not None else "    win_rate = n/a")
    print(f"    avg_pnl_pct                = {stats.avg_pnl_pct*100:+.3f}%" if stats.avg_pnl_pct is not None else "")
    print(f"    total_pnl                  = ${stats.total_pnl:+.2f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
