"""Persistent record of REALIZED round-trip trades from live/paper auto-trading.

Separate from execution_log.py, which only logs raw order-submission attempts
(no strategy attribution, no round-trip/P&L concept). This is the ground truth
the adaptive roster's demotion rules read from — real trade outcomes, not
backtest simulation.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "live_trades.db"

# Strategy name recorded when a closed round trip has no attribution — the
# position predates the attribution system, was opened manually, or its record
# was pruned. Such trades USED to be dropped entirely, silently understating
# realised P&L (found 2026-08-14: an orphaned DDOG close cost -$0.07 that
# never appeared anywhere). Recording them under this placeholder keeps the
# books honest while deliberately NOT counting toward any real strategy's
# recent_performance(), so an unprovable trade can't drive a demotion.
UNATTRIBUTED_STRATEGY = "(unattributed)"

SCHEMA = """
CREATE TABLE IF NOT EXISTS live_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    account_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    is_paper INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    qty REAL NOT NULL,
    pnl REAL NOT NULL,
    pnl_pct REAL NOT NULL,
    conviction REAL
);

CREATE INDEX IF NOT EXISTS idx_live_trades_ticker_strategy ON live_trades(ticker, strategy_name);
CREATE INDEX IF NOT EXISTS idx_live_trades_account ON live_trades(account_id);
"""


@dataclass
class PerformanceSnapshot:
    num_trades: int
    win_rate: float | None
    avg_pnl_pct: float | None
    total_pnl: float
    current_losing_streak: int
    max_pnl_drawdown: float  # $ peak-to-trough of cumulative pnl over the lookback window —
    # NOT comparable to a backtest's normalized max_drawdown (different units).


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn) -> None:
    """Add columns to a pre-existing live_trades table that predates them.
    CREATE TABLE IF NOT EXISTS won't alter an existing table. Idempotent (guarded
    by a table_info check) — safe to run on every startup."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(live_trades)").fetchall()}
    if "conviction" not in existing:
        conn.execute("ALTER TABLE live_trades ADD COLUMN conviction REAL")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def record_realized_trade(
    account_id: str,
    ticker: str,
    strategy_name: str,
    is_paper: bool,
    entry_time: str,
    entry_price: float,
    exit_time: str,
    exit_price: float,
    qty: float,
    conviction: float | None = None,
) -> int:
    """Record one closed round trip (long-only: always BUY then SELL, matching
    this project's engine.py convention). `conviction` is the [0,1] strength of
    the ENTRY signal (#25), captured at open time and carried through — logged
    only, never used to size. Returns the new row id."""
    pnl = (exit_price - entry_price) * qty
    pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0

    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO live_trades
               (account_id, ticker, strategy_name, is_paper, entry_time, entry_price,
                exit_time, exit_price, qty, pnl, pnl_pct, conviction)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                account_id,
                ticker,
                strategy_name,
                int(is_paper),
                entry_time,
                entry_price,
                exit_time,
                exit_price,
                qty,
                pnl,
                pnl_pct,
                conviction,
            ),
        )
        row_id = cursor.lastrowid
    return row_id


def list_recent_trades(limit: int = 50, account_id: str | None = None) -> list[dict]:
    """Most recent closed round trips across every (ticker, strategy) combo,
    newest first — for the mobile app's trade-history view (#31). Excludes
    the literal 'mock-1' account_id: early-development test/mock rows, not a
    real linked account (see broker_accounts.json for what those look like —
    real UUIDs, not a human-readable placeholder), which would be actively
    misleading shown in a display of real trading activity rather than a
    display nuance worth silently tolerating.

    account_id: restrict to one account — what the app's per-account
    "closed positions" list needs. account_id is also RETURNED now (it was
    always selected out before), so a combined view can group by account
    without a second lookup.
    """
    init_db()
    where = "account_id != 'mock-1'"
    params: list = []
    if account_id is not None:
        where += " AND account_id = ?"
        params.append(account_id)
    params.append(limit)

    with _connect() as conn:
        rows = conn.execute(
            f"""SELECT account_id, ticker, strategy_name, is_paper, entry_time, entry_price,
                       exit_time, exit_price, qty, pnl, pnl_pct, conviction
                FROM live_trades
                WHERE {where}
                ORDER BY exit_time DESC LIMIT ?""",
            tuple(params),
        ).fetchall()

    columns = [
        "account_id", "ticker", "strategy_name", "is_paper", "entry_time", "entry_price",
        "exit_time", "exit_price", "qty", "pnl", "pnl_pct", "conviction",
    ]
    trades = []
    for row in rows:
        trade = dict(zip(columns, row))
        trade["is_paper"] = bool(trade["is_paper"])
        trades.append(trade)
    return trades


def realized_pnl_by_account() -> dict[str, float]:
    """Sum of realized P&L (closed round trips only, not unrealized/open-position
    P&L) grouped by account_id — for the Accounts tab's per-account and grand
    total figures. Excludes 'mock-1' for the same reason as list_recent_trades.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT account_id, SUM(pnl) FROM live_trades
               WHERE account_id != 'mock-1'
               GROUP BY account_id"""
        ).fetchall()
    return {account_id: total for account_id, total in rows}


def realized_pnl_by_account_between(start_iso: str, end_iso: str) -> dict[str, float]:
    """Same shape as realized_pnl_by_account(), scoped to trades whose exit_time
    falls in [start_iso, end_iso] — for the Deposits section's weekly
    deposit-vs-trading-growth breakdown. Excludes 'mock-1' for the same reason
    as list_recent_trades.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT account_id, SUM(pnl) FROM live_trades
               WHERE account_id != 'mock-1' AND exit_time BETWEEN ? AND ?
               GROUP BY account_id""",
            (start_iso, end_iso),
        ).fetchall()
    return {account_id: total for account_id, total in rows}


# A single trading DECISION (the roster picking a combo and opening/closing
# it) produces one live_trades row PER CONNECTED ACCOUNT — auto_trader loops
# over every target account in one cycle and submits to each. Observed live
# (2026-08-14, Q/VWAP Mean Reversion): three accounts closed the same decision
# at 16:18:47, 16:18:47, 16:18:46 — within a second of each other. The next
# poll cycle is poll_interval_seconds away (60s by default), so a tolerance
# this wide has no realistic way to merge two genuinely separate decisions.
DECISION_GROUP_SECONDS = 10.0

# Raw rows fetched per attempt while grouping toward lookback_n decisions, and
# the hard ceiling on how far that doubles — bounds one status check to a
# handful of queries even if an unusual number of accounts (or a very
# long-tenured combo) means many rows are needed to reach lookback_n decisions.
_DECISION_FETCH_START_MULTIPLIER = 3
_DECISION_FETCH_CAP = 500


def _group_into_decisions(rows_chrono: list[tuple[str, float, float]]) -> list[dict]:
    """rows_chrono = [(exit_time_iso, pnl, pnl_pct), ...], oldest first, for
    ONE (ticker, strategy) — already scoped by the caller's WHERE clause, so
    grouping here only needs to reason about time proximity.

    Collapses rows within DECISION_GROUP_SECONDS of the group's FIRST row
    (not a sliding window off the previous row, which would let a slow
    sequence of near-but-not-quite-simultaneous rows chain-drift into one
    false decision) into a single entry:
      pnl      - SUM of dollar pnl across the grouped rows. Real money moved
                 on every account; summing it is not double-counting, it is
                 the correct total for what this decision actually made.
      pnl_pct  - MEAN pnl_pct across the grouped rows. Deliberately NOT a
                 dollar figure — a decision that gains 0.09% on average but
                 shows a dollar LOSS on the biggest connected account (real
                 case: Q/VWAP MR 2026-08-14 13:42, IBKR Paper +$19.59, MyAlpaca
                 -$5.54, mean +0.091%) should count as the win it was, not a
                 loss dictated by whichever account happens to be largest.
      is_loss / is_win - derived from pnl_pct, mirroring the exact tri-state
                 the un-grouped code used on raw pnl: strictly negative is a
                 loss (continues a losing streak), strictly positive is a win
                 (counted in win_rate), exactly zero is neither.
    """
    decisions: list[dict] = []
    group: list[tuple[str, float, float]] = []
    anchor: datetime | None = None

    def flush() -> None:
        if not group:
            return
        pnls = [r[1] for r in group]
        pcts = [r[2] for r in group]
        mean_pct = sum(pcts) / len(pcts)
        decisions.append({
            "exit_time": group[-1][0],
            "pnl": sum(pnls),
            "pnl_pct": mean_pct,
            "is_loss": mean_pct < 0,
            "is_win": mean_pct > 0,
        })

    for exit_time, pnl, pnl_pct in rows_chrono:
        ts = datetime.fromisoformat(exit_time.replace("Z", "+00:00"))
        if anchor is not None and (ts - anchor).total_seconds() > DECISION_GROUP_SECONDS:
            flush()
            group = []
            anchor = None
        group.append((exit_time, pnl, pnl_pct))
        anchor = anchor or ts
    flush()
    return decisions


def recent_performance(ticker: str, strategy_name: str, lookback_n: int = 20) -> PerformanceSnapshot:
    """Rolling performance for one (ticker, strategy) combo from its most
    recent `lookback_n` closed trading DECISIONS — not raw live_trades rows.
    num_trades=0 when nothing has closed yet for this combo.

    A decision closed across N connected accounts writes N rows; counting rows
    directly overcounted every real outcome by the number of accounts trading
    it, which meant a 3-account combo's "losing_streak_threshold=3" rule was
    really a 1-strike rule, and its 20-trade lookback window was really about
    7 decisions. See _group_into_decisions for the grouping itself and the
    real incident that surfaced this (Q/VWAP Mean Reversion paused 2026-08-14
    on what was, decision-wise, a single loss).
    """
    init_db()
    fetch_limit = lookback_n * _DECISION_FETCH_START_MULTIPLIER
    decisions_chrono: list[dict] = []
    while True:
        with _connect() as conn:
            rows_desc = conn.execute(
                """SELECT exit_time, pnl, pnl_pct FROM live_trades
                   WHERE ticker = ? AND strategy_name = ?
                   ORDER BY exit_time DESC LIMIT ?""",
                (ticker, strategy_name, fetch_limit),
            ).fetchall()
        if not rows_desc:
            return PerformanceSnapshot(
                num_trades=0,
                win_rate=None,
                avg_pnl_pct=None,
                total_pnl=0.0,
                current_losing_streak=0,
                max_pnl_drawdown=0.0,
            )
        decisions_chrono = _group_into_decisions(list(reversed(rows_desc)))
        # Stop once there's enough material for the requested lookback, or
        # once fetching more rows genuinely can't produce more decisions —
        # either this combo has no earlier history, or the fetch is already
        # capped. Otherwise widen the fetch and try again.
        if (
            len(decisions_chrono) >= lookback_n
            or len(rows_desc) < fetch_limit
            or fetch_limit >= _DECISION_FETCH_CAP
        ):
            break
        fetch_limit = min(fetch_limit * 2, _DECISION_FETCH_CAP)

    decisions_chrono = decisions_chrono[-lookback_n:]
    decisions_desc = list(reversed(decisions_chrono))

    current_losing_streak = 0
    for d in decisions_desc:
        if d["is_loss"]:
            current_losing_streak += 1
        else:
            break

    running_total = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for d in decisions_chrono:
        running_total += d["pnl"]
        peak = max(peak, running_total)
        max_drawdown = max(max_drawdown, peak - running_total)

    num_trades = len(decisions_chrono)
    win_rate = sum(1 for d in decisions_chrono if d["is_win"]) / num_trades
    avg_pnl_pct = sum(d["pnl_pct"] for d in decisions_chrono) / num_trades
    total_pnl = sum(d["pnl"] for d in decisions_chrono)

    return PerformanceSnapshot(
        num_trades=num_trades,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl_pct,
        total_pnl=total_pnl,
        current_losing_streak=current_losing_streak,
        max_pnl_drawdown=max_drawdown,
    )
