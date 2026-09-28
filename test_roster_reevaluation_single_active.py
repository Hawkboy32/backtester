"""Regression test for a real bug found live 2026-09-28: when a ticker
already active under one strategy gets re-promoted under a DIFFERENT,
higher-scoring strategy in the same evaluate_roster() pass, the OLD combo
must be demoted, not silently left "active" too. evaluate_roster's own
docstring promises "at most one active strategy per ticker" - this tests
that promise directly, since getting it wrong means real live trading on
two simultaneous combos for one ticker (found live: ECHO ended up active
under both Linear Regression Channel and VWAP Mean Reversion at once).

    .venv/Scripts/python test_roster_reevaluation_single_active.py
"""
from __future__ import annotations

import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import roster  # noqa: E402
from backtester.live_trades import PerformanceSnapshot  # noqa: E402
from backtester.scanner import ScanResultRow  # noqa: E402


def _zero_trades(ticker: str, strategy_name: str) -> PerformanceSnapshot:
    # num_trades=0 keeps _demotion_reason a no-op, so this test isolates the
    # re-ranking/claiming bug from the (unrelated) live-performance rules.
    return PerformanceSnapshot(
        num_trades=0, win_rate=None, avg_pnl_pct=None, total_pnl=0.0,
        current_losing_streak=0, max_pnl_drawdown=0.0,
    )


def _make_config() -> roster.RosterConfig:
    cfg = roster.RosterConfig()
    cfg.roster_size = 6
    cfg.max_per_strategy = 6  # generous - isolate the claiming bug from diversification caps
    cfg.max_per_sector = 6
    cfg.regime_match_only = False
    return cfg


def test_reevaluation_demotes_the_superseded_combo_for_the_same_ticker():
    now = datetime.now(timezone.utc).isoformat()
    current_roster = roster.RosterState(
        entries=[
            roster.RosterEntry(
                ticker="ECHO", strategy_name="Old Strategy", status="active",
                promoted_at=now, backtest_score=0.5,
            ),
        ],
        config=_make_config(),
    )

    scan_rows = [
        # Old Strategy scores lower now - this is what should get demoted.
        ScanResultRow(
            ticker="ECHO", strategy_name="Old Strategy", params={},
            sharpe_ratio=0.5, total_return=0.05, max_drawdown=-0.05, win_rate=0.5,
        ),
        # New Strategy scores higher - this is what should get promoted.
        ScanResultRow(
            ticker="ECHO", strategy_name="New Strategy", params={},
            sharpe_ratio=3.0, total_return=0.30, max_drawdown=-0.02, win_rate=0.8,
        ),
    ]

    new_state = roster.evaluate_roster(
        scan_rows, _zero_trades, current_roster, config=current_roster.config, dry_run=True,
    )

    echo_entries = [e for e in new_state.entries if e.ticker == "ECHO"]
    active_echo = [e for e in echo_entries if e.status == "active"]

    assert len(echo_entries) == 2, f"expected both ECHO combos present (one active, one demoted), got {echo_entries}"
    assert len(active_echo) == 1, f"ECHO must have EXACTLY one active combo, got {len(active_echo)}: {active_echo}"
    assert active_echo[0].strategy_name == "New Strategy", \
        f"the higher-scored combo should be the active one, got {active_echo[0].strategy_name}"

    old = next(e for e in echo_entries if e.strategy_name == "Old Strategy")
    assert old.status == "candidate", f"the superseded combo must be demoted, not left {old.status!r}"


def test_no_ticker_ever_ends_up_with_more_than_one_active_combo():
    # Broader invariant check, not just the ECHO case - covers 3+ candidate
    # strategies for one ticker too.
    now = datetime.now(timezone.utc).isoformat()
    current_roster = roster.RosterState(
        entries=[
            roster.RosterEntry(ticker="XYZ", strategy_name="A", status="active", promoted_at=now, backtest_score=0.4),
        ],
        config=_make_config(),
    )
    scan_rows = [
        ScanResultRow(ticker="XYZ", strategy_name="A", params={}, sharpe_ratio=0.4, total_return=0.04, max_drawdown=-0.05, win_rate=0.5),
        ScanResultRow(ticker="XYZ", strategy_name="B", params={}, sharpe_ratio=2.0, total_return=0.20, max_drawdown=-0.03, win_rate=0.7),
        ScanResultRow(ticker="XYZ", strategy_name="C", params={}, sharpe_ratio=1.0, total_return=0.10, max_drawdown=-0.04, win_rate=0.6),
    ]

    new_state = roster.evaluate_roster(
        scan_rows, _zero_trades, current_roster, config=current_roster.config, dry_run=True,
    )

    active_counts = Counter(e.ticker for e in new_state.entries if e.status == "active")
    over_claimed = {t: c for t, c in active_counts.items() if c > 1}
    assert not over_claimed, f"tickers with more than one active combo: {over_claimed}"


if __name__ == "__main__":
    test_reevaluation_demotes_the_superseded_combo_for_the_same_ticker()
    test_no_ticker_ever_ends_up_with_more_than_one_active_combo()
    print("All tests passed.")
