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


def list_recent_trades(limit: int = 50) -> list[dict]:
    """Most recent closed round trips across every (ticker, strategy) combo,
    newest first — for the mobile app's trade-history view (#31). Excludes
    the literal 'mock-1' account_id: early-development test/mock rows, not a
    real linked account (see broker_accounts.json for what those look like —
    real UUIDs, not a human-readable placeholder), which would be actively
    misleading shown in a display of real trading activity rather than a
    display nuance worth silently tolerating.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ticker, strategy_name, is_paper, entry_time, entry_price,
                      exit_time, exit_price, qty, pnl, pnl_pct, conviction
               FROM live_trades
               WHERE account_id != 'mock-1'
               ORDER BY exit_time DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    columns = [
        "ticker", "strategy_name", "is_paper", "entry_time", "entry_price",
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


def recent_performance(ticker: str, strategy_name: str, lookback_n: int = 20) -> PerformanceSnapshot:
    """Rolling performance for one (ticker, strategy) combo from its most
    recent `lookback_n` closed live/paper trades, oldest-first internally for
    the drawdown calc. num_trades=0 when nothing has closed yet for this combo.
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT exit_time, pnl, pnl_pct FROM live_trades
               WHERE ticker = ? AND strategy_name = ?
               ORDER BY exit_time DESC LIMIT ?""",
            (ticker, strategy_name, lookback_n),
        ).fetchall()

    if not rows:
        return PerformanceSnapshot(
            num_trades=0,
            win_rate=None,
            avg_pnl_pct=None,
            total_pnl=0.0,
            current_losing_streak=0,
            max_pnl_drawdown=0.0,
        )

    # rows are DESC (newest first) — use as-is for the losing-streak walk,
    # then reverse to chronological order for the cumulative-pnl drawdown.
    pnls_desc = [r[1] for r in rows]
    pnl_pcts = [r[2] for r in rows]

    current_losing_streak = 0
    for pnl in pnls_desc:
        if pnl < 0:
            current_losing_streak += 1
        else:
            break

    pnls_chrono = list(reversed(pnls_desc))
    running_total = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls_chrono:
        running_total += pnl
        peak = max(peak, running_total)
        max_drawdown = max(max_drawdown, peak - running_total)

    num_trades = len(rows)
    win_rate = sum(1 for p in pnls_desc if p > 0) / num_trades
    avg_pnl_pct = sum(pnl_pcts) / num_trades
    total_pnl = sum(pnls_desc)

    return PerformanceSnapshot(
        num_trades=num_trades,
        win_rate=win_rate,
        avg_pnl_pct=avg_pnl_pct,
        total_pnl=total_pnl,
        current_losing_streak=current_losing_streak,
        max_pnl_drawdown=max_drawdown,
    )
