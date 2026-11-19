"""Read-only report on the opening-minutes rule (see src/backtester/entry_window.py).

    .venv/Scripts/python entry_block_report.py [--since YYYY-MM-DD]

Reads auto_trader_state/entry_blocks.jsonl (written by the trader) and the closed
real-money trades in live_trades.db (opened read-only). Writes nothing.

It answers three questions:
  1. How many entries did the rule hold back, per day?
  2. Did any equity entry slip through the window anyway? (should be none)
  3. Was a held-back signal just DELAYED - the same strategy on the same ticker
     entering later that day - or really skipped? (the lab's test skipped such
     entries; a rule that only delays them is a weaker version of it)

It cannot say whether the rule is paying off: that needs months of fills.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from backtester import entry_window as ew  # noqa: E402
from backtester.accounts import infer_asset_class  # noqa: E402

NY = ZoneInfo("America/New_York")
RULE_LIVE_FROM = "2026-11-18"  # the trader was restarted onto the rule on this day (UTC date)
DB_PATH = HERE / "live_trades.db"


def parse_ts(text: str) -> datetime:
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def read_blocks(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(raw))
        except json.JSONDecodeError:
            continue  # a half-written last line is not worth stopping for
    return rows


def read_trades(db_path: Path, since_utc_date: str) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        cur = con.execute(
            "select account_id, ticker, strategy_name, is_paper, entry_time, pnl_pct from live_trades "
            "where entry_time >= ? and is_paper = 0",
            (since_utc_date,),
        )
        names = ["account_id", "ticker", "strategy_name", "is_paper", "entry_time", "pnl_pct"]
        return [dict(zip(names, r)) for r in cur]
    finally:
        con.close()


def build_report(blocks: list[dict], trades: list[dict], skip_minutes: int, since: str) -> dict:
    blocks = [b for b in blocks if b["ny_time"][:10] >= since]

    per_day: dict[str, int] = defaultdict(int)
    for b in blocks:
        per_day[b["ny_time"][:10]] += 1

    # real-money equity entries made inside the window since the rule went live
    leaks = []
    equity_entries = 0
    for t in trades:
        if infer_asset_class(t["ticker"]) != "equity":
            continue
        entered = parse_ts(t["entry_time"])
        if entered.astimezone(NY).date().isoformat() < since:
            continue
        equity_entries += 1
        if ew.in_opening_window(entered, skip_minutes):
            leaks.append(t)

    # delayed or skipped: a later same-day entry by the same strategy on the same ticker and account
    delayed = 0
    skipped = 0
    for b in blocks:
        blocked_at = parse_ts(b["timestamp"])
        day = b["ny_time"][:10]
        later = any(
            t["account_id"] == b["account_id"]
            and t["ticker"] == b["ticker"]
            and t["strategy_name"] == b["strategy_name"]
            and parse_ts(t["entry_time"]) > blocked_at
            and parse_ts(t["entry_time"]).astimezone(NY).date().isoformat() == day
            for t in trades
        )
        delayed += later
        skipped += not later
    return {
        "since": since,
        "blocked": len(blocks),
        "per_day": dict(sorted(per_day.items())),
        "equity_entries": equity_entries,
        "leaks": leaks,
        "delayed": delayed,
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--since", default=RULE_LIVE_FROM, help="first New York date to count (default: the day the rule went live)")
    ap.add_argument("--skip-minutes", type=int, default=15, help="window length the trader was running with")
    args = ap.parse_args(argv)

    blocks_path = ew.BLOCKS_PATH
    blocks = read_blocks(blocks_path)
    trades = read_trades(DB_PATH, args.since)
    r = build_report(blocks, trades, args.skip_minutes, args.since)

    print(f"Opening-minutes rule report, from {r['since']} (New York dates)")
    if not blocks_path.exists():
        print("  No entry_blocks.jsonl yet: the trader has not been restarted onto the logging build, "
              "or no entry has been held back since.")
    print(f"  Entries held back: {r['blocked']}")
    for day, n in r["per_day"].items():
        print(f"    {day}: {n}")
    print(f"  Real-money equity entries since then: {r['equity_entries']}; "
          f"inside the {args.skip_minutes}-minute window: {len(r['leaks'])} (should be 0)")
    for t in r["leaks"]:
        print(f"    LEAK  {t['entry_time']}  {t['ticker']}  {t['strategy_name']}")
    if r["blocked"]:
        print(f"  Of the held-back entries, entered later the same day (delayed, not skipped): {r['delayed']}; "
              f"no later entry (really skipped): {r['skipped']}")
    print("  This cannot say whether the rule helps: that needs months of fills.")
    return 1 if r["leaks"] else 0


if __name__ == "__main__":
    sys.exit(main())
