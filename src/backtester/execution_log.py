"""Persistent, append-only log of every order execution attempt.

Separate from the in-session results table shown right after a submit —
this survives across dashboard restarts so you can reconcile what actually
got sent to which account over time.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from backtester.brokers.base import OrderResult, OrderSide

LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "execution_log.jsonl"


def log_results(ticker: str, side: OrderSide, results: list[OrderResult]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as f:
        for r in results:
            entry = {"timestamp": timestamp, "ticker": ticker, "side": side.value, **asdict(r)}
            f.write(json.dumps(entry) + "\n")


def load_log(limit: int | None = 200) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame(
            columns=["timestamp", "ticker", "side", "account_nickname", "success", "broker_order_id", "error"]
        )
    rows = []
    with LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp", ascending=False)
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)
