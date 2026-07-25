"""Persistent, accumulating store for Scanner results.

Every scan run gets appended here — nothing is overwritten, nothing needs
downloading to be seen again. bot_memory.txt (see memory_report.py) still
gets written for anyone who wants a standalone export, but it's no longer
the only way to see results: query this database directly instead.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtester.scanner import ScanResultRow

DB_PATH = Path(__file__).resolve().parent.parent.parent / "scan_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    universe TEXT,
    num_tickers INTEGER,
    from_date TEXT,
    to_date TEXT,
    multiplier INTEGER,
    timespan TEXT,
    strategy_names TEXT
);

CREATE TABLE IF NOT EXISTS scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES scan_runs(id),
    ticker TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    params TEXT,
    total_return REAL,
    cagr REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    num_trades INTEGER,
    win_rate REAL,
    num_bars INTEGER,
    conviction REAL,
    expectancy REAL,
    efficiency_ratio REAL,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_results_run ON scan_results(run_id);
CREATE INDEX IF NOT EXISTS idx_results_ticker ON scan_results(ticker);
CREATE INDEX IF NOT EXISTS idx_results_strategy ON scan_results(strategy_name);
"""


@contextmanager
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def _migrate(conn) -> None:
    """Bring an older scan_results table up to date. CREATE TABLE IF NOT EXISTS
    won't add columns to a pre-existing table, so add any missing ones here.
    Idempotent (guarded by a table_info check) — safe to run on every startup."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(scan_results)").fetchall()}
    if "conviction" not in existing:
        conn.execute("ALTER TABLE scan_results ADD COLUMN conviction REAL")
    if "expectancy" not in existing:
        conn.execute("ALTER TABLE scan_results ADD COLUMN expectancy REAL")
    if "efficiency_ratio" not in existing:
        conn.execute("ALTER TABLE scan_results ADD COLUMN efficiency_ratio REAL")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def record_scan(meta: dict, rows: list[ScanResultRow]) -> int:
    """Persist one scan run and all its result rows. Returns the run id."""
    init_db()
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO scan_runs
               (run_at, universe, num_tickers, from_date, to_date, multiplier, timespan, strategy_names)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                meta.get("run_at", datetime.now(timezone.utc).isoformat()),
                meta.get("universe"),
                meta.get("num_tickers"),
                meta.get("from_date"),
                meta.get("to_date"),
                meta.get("multiplier"),
                meta.get("timespan"),
                json.dumps(meta.get("strategy_names", [])),
            ),
        )
        run_id = cursor.lastrowid

        conn.executemany(
            """INSERT INTO scan_results
               (run_id, ticker, strategy_name, params, total_return, cagr, max_drawdown,
                sharpe_ratio, num_trades, win_rate, num_bars, conviction, expectancy,
                efficiency_ratio, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    run_id,
                    r.ticker,
                    r.strategy_name,
                    json.dumps(r.params),
                    r.total_return,
                    r.cagr,
                    r.max_drawdown,
                    r.sharpe_ratio,
                    r.num_trades,
                    r.win_rate,
                    r.num_bars,
                    r.avg_conviction,
                    r.expectancy,
                    r.efficiency_ratio,
                    r.error,
                )
                for r in rows
            ],
        )
    return run_id


def query_summary_counts() -> dict:
    init_db()
    with _connect() as conn:
        num_runs = conn.execute("SELECT COUNT(*) FROM scan_runs").fetchone()[0]
        num_results = conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0]
        num_tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM scan_results").fetchone()[0]
        num_strategies = conn.execute(
            "SELECT COUNT(DISTINCT strategy_name) FROM scan_results"
        ).fetchone()[0]
    return {
        "num_runs": num_runs,
        "num_results": num_results,
        "num_tickers": num_tickers,
        "num_strategies": num_strategies,
    }


def query_latest_run_id() -> int | None:
    init_db()
    with _connect() as conn:
        row = conn.execute("SELECT id FROM scan_runs ORDER BY id DESC LIMIT 1").fetchone()
    return row[0] if row else None


def query_run_results(run_id: int) -> list[ScanResultRow]:
    """Raw ScanResultRow objects for one run — for feeding back into
    ranking.rank_combos() (e.g. from the Auto Trading tab's adaptive roster,
    which needs actual rows, not the DataFrame-shaped query functions below).
    """
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            """SELECT ticker, strategy_name, params, total_return, cagr, max_drawdown,
                      sharpe_ratio, num_trades, win_rate, num_bars, conviction,
                      expectancy, efficiency_ratio, error
               FROM scan_results WHERE run_id = ?""",
            (run_id,),
        ).fetchall()

    return [
        ScanResultRow(
            ticker=r[0],
            strategy_name=r[1],
            params=json.loads(r[2]) if r[2] else {},
            total_return=r[3],
            cagr=r[4],
            max_drawdown=r[5],
            sharpe_ratio=r[6],
            num_trades=r[7],
            win_rate=r[8],
            num_bars=r[9],
            avg_conviction=r[10],
            expectancy=r[11],
            efficiency_ratio=r[12],
            error=r[13],
        )
        for r in rows
    ]


def query_recent_runs(limit: int = 20) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query(
            "SELECT * FROM scan_runs ORDER BY id DESC LIMIT ?", conn, params=(limit,)
        )
    return df


def query_all_time_top(
    limit: int = 30, ticker: str | None = None, strategy_name: str | None = None
) -> pd.DataFrame:
    """Every non-error result ever recorded, ranked by Sharpe ratio, optionally filtered."""
    init_db()
    clauses = ["error IS NULL"]
    params: list = []
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker)
    if strategy_name:
        clauses.append("strategy_name = ?")
        params.append(strategy_name)
    where = " AND ".join(clauses)
    params.append(limit)

    with _connect() as conn:
        df = pd.read_sql_query(
            f"""SELECT sr.ticker, sr.strategy_name, sr.total_return, sr.cagr, sr.max_drawdown,
                       sr.sharpe_ratio, sr.num_trades, sr.win_rate, sr.conviction,
                       sr.expectancy, sr.efficiency_ratio, sc.run_at
                FROM scan_results sr
                JOIN scan_runs sc ON sc.id = sr.run_id
                WHERE {where}
                ORDER BY sr.sharpe_ratio DESC
                LIMIT ?""",
            conn,
            params=params,
        )
    return df


def query_strategy_leaderboard() -> pd.DataFrame:
    """Per-strategy performance aggregated across every run ever recorded."""
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query(
            """SELECT
                   strategy_name,
                   COUNT(*) AS num_results,
                   COUNT(DISTINCT run_id) AS num_runs,
                   AVG(sharpe_ratio) AS mean_sharpe,
                   AVG(total_return) AS mean_return,
                   AVG(CASE WHEN total_return > 0 THEN 1.0 ELSE 0.0 END) AS pct_profitable,
                   AVG(conviction) AS mean_conviction,
                   AVG(expectancy) AS mean_expectancy
               FROM scan_results
               WHERE error IS NULL
               GROUP BY strategy_name
               ORDER BY mean_sharpe DESC""",
            conn,
        )
    return df


def query_ticker_history(ticker: str) -> pd.DataFrame:
    init_db()
    with _connect() as conn:
        df = pd.read_sql_query(
            """SELECT sr.strategy_name, sr.total_return, sr.sharpe_ratio, sr.max_drawdown,
                      sr.num_trades, sr.win_rate, sc.run_at
               FROM scan_results sr
               JOIN scan_runs sc ON sc.id = sr.run_id
               WHERE sr.ticker = ? AND sr.error IS NULL
               ORDER BY sc.run_at DESC""",
            conn,
            params=(ticker,),
        )
    return df


def clear_history() -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM scan_results")
        conn.execute("DELETE FROM scan_runs")
