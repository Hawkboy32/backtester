"""The opening-minutes rule (added 2026-11-17 from the strategy lab's Study I-H): no NEW equity entry in the first skip_opening_minutes after 09:30 New York.

Opening only: a close, crypto, forex (and so the IG index Opening Spike Fade) are never held back.

    .venv/Scripts/python test_entry_window.py
"""
from __future__ import annotations

import ast
import json
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from backtester import entry_window as ew  # noqa: E402
from backtester.auto_trader_state import AutoTraderControl, from_dict  # noqa: E402

PASS, FAIL = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -  {detail}" if detail and not ok else ""))


def utc(y, mo, d, h, mi, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=timezone.utc)


def _readonly_refuses(db: Path) -> bool:
    """A connection opened the way the report opens it (mode=ro) must refuse a write."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        con.execute("insert into live_trades (ticker) values ('X')")
        return False
    except sqlite3.OperationalError:
        return True
    finally:
        con.close()


def main() -> int:
    # --- the window edges, in winter (EST = UTC-5): 09:30 NY is 14:30 UTC. 2026-03-02 is a Monday.
    w = lambda h, m, s=0: ew.in_opening_window(utc(2026, 3, 2, h, m, s), 15)
    check("winter: 09:29:59 is NOT in the window", not w(14, 29, 59))
    check("winter: 09:30:00 IS in the window", w(14, 30, 0))
    check("winter: 09:44:59 IS in the window", w(14, 44, 59))
    check("winter: 09:45:00 is NOT in the window", not w(14, 45, 0))
    check("winter: mid-session and pre-market are not in the window", not w(17, 0) and not w(10, 0))

    # --- summer (EDT = UTC-4): 09:30 NY is 13:30 UTC, so the UTC hour moves; 2026-07-01 is a Wednesday.
    s = lambda h, m, sec=0: ew.in_opening_window(utc(2026, 7, 1, h, m, sec), 15)
    check("summer: 13:30 UTC (09:30 EDT) IS in the window", s(13, 30))
    check("summer: 13:44:59 UTC IS in the window", s(13, 44, 59))
    check("summer: 13:45 UTC is NOT in the window", not s(13, 45))
    check("summer: 14:30 UTC (10:30 EDT) is NOT in the window (a fixed-UTC rule would be wrong here)", not s(14, 30))
    check("summer: 13:29:59 UTC is NOT in the window", not s(13, 29, 59))

    # --- the two daylight-saving changeovers are weekends in the US, so no weekday ever straddles them;
    # the Monday after spring-forward (2026-03-09, EDT) must already use UTC-4
    check("Monday after the clocks go forward: 13:30 UTC is the open", ew.in_opening_window(utc(2026, 3, 9, 13, 30), 15) and not ew.in_opening_window(utc(2026, 3, 9, 14, 30), 15))
    check("Monday after the clocks go back (2026-11-02, EST): 14:30 UTC is the open", ew.in_opening_window(utc(2026, 11, 2, 14, 30), 15) and not ew.in_opening_window(utc(2026, 11, 2, 13, 30), 15))

    # --- weekends never
    check("Saturday 14:30 UTC is not in the window", not ew.in_opening_window(utc(2026, 3, 7, 14, 30), 15))
    check("Sunday 14:30 UTC is not in the window", not ew.in_opening_window(utc(2026, 3, 8, 14, 30), 15))

    # --- the length setting
    check("0 turns it off", not ew.in_opening_window(utc(2026, 3, 2, 14, 30), 0))
    check("a negative value turns it off", not ew.in_opening_window(utc(2026, 3, 2, 14, 30), -5))
    check("a longer window (30) covers 09:59:59 and not 10:00", ew.in_opening_window(utc(2026, 3, 2, 14, 59, 59), 30) and not ew.in_opening_window(utc(2026, 3, 2, 15, 0), 30))
    check("a typo-sized window is capped at 60 minutes", ew.in_opening_window(utc(2026, 3, 2, 15, 29), 9999) and not ew.in_opening_window(utc(2026, 3, 2, 15, 30), 9999))
    for bad in (None, "15", 15.0, True, [15]):
        check(f"a bad value {bad!r} fails open (never blocks on garbage)", not ew.in_opening_window(utc(2026, 3, 2, 14, 30), bad))

    # --- what is held back: opening equity only
    now = utc(2026, 3, 2, 14, 31)
    check("an OPENING equity order in the window is held back (with a reason)", isinstance(ew.opening_block_reason("equity", True, now, 15), str))
    check("a CLOSING equity order in the window is NEVER held back", ew.opening_block_reason("equity", False, now, 15) is None)
    check("crypto opening in the window is not held back", ew.opening_block_reason("crypto", True, now, 15) is None)
    check("forex opening in the window is not held back (this is also the IG index strategy)", ew.opening_block_reason("forex", True, now, 15) is None)
    check("an opening equity order after the window is not held back", ew.opening_block_reason("equity", True, utc(2026, 3, 2, 14, 45), 15) is None)
    check("with the setting at 0 nothing is held back", ew.opening_block_reason("equity", True, now, 0) is None)

    # --- the setting itself
    check("the default is 15", AutoTraderControl().skip_opening_minutes == 15)
    old_control = {"enabled": True, "killed": False}  # a control.json written before the field existed
    check("an old control.json without the field loads with 15", from_dict(AutoTraderControl, old_control).skip_opening_minutes == 15)
    check("a stored 0 is kept as 0", from_dict(AutoTraderControl, {"skip_opening_minutes": 0}).skip_opening_minutes == 0)

    # --- the wiring: the trader calls the rule, on the opening branch only, before any order is queued
    src = (HERE / "auto_trader.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_trade_target")
    seg = ast.get_source_segment(src, fn)
    check("auto_trader imports entry_window", "entry_window" in src.split("def _get_broker_accounts")[0])
    i_rule = seg.find("entry_window.opening_block_reason(")
    i_queue = seg.find("account_orders.append(")
    i_close = seg.find("if not is_opening:")
    check("_trade_target calls the rule", i_rule != -1)
    check("the rule runs before the order is queued", -1 < i_rule < i_queue)
    check("the rule runs before the sizing branch", -1 < i_rule < i_close)
    check("the rule is given is_opening (so a close is never blocked), the asset class and the setting",
          "ticker_asset_class, is_opening, datetime.now(timezone.utc), control.skip_opening_minutes" in seg)
    check("a held-back entry skips with `continue`, like the event and storm blocks",
          "opening_reason is not None" in seg and seg[seg.find("opening_reason is not None"):].split("continue", 1)[0].count("\n") < 10)
    app_src = (HERE / "app.py").read_text(encoding="utf-8")
    check("the dashboard's Save carries the setting forward (so saving does not reset it)", "skip_opening_minutes=control.skip_opening_minutes" in app_src)

    # --- the block log (entry_blocks.jsonl): one line per account/ticker/strategy/side/day, never breaks trading
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "entry_blocks.jsonl"
        seen: set = set()
        t0 = utc(2026, 3, 2, 14, 31)
        first = ew.record_block(t0, "acct1", "AAPL", "VWAP Mean Reversion", "buy", 15, path=log, seen=seen)
        again = ew.record_block(utc(2026, 3, 2, 14, 32), "acct1", "AAPL", "VWAP Mean Reversion", "buy", 15, path=log, seen=seen)
        check("the first block is written and the repeat on the next poll is not", first and not again)
        ew.record_block(t0, "acct2", "AAPL", "VWAP Mean Reversion", "buy", 15, path=log, seen=seen)
        ew.record_block(t0, "acct1", "MSFT", "VWAP Mean Reversion", "buy", 15, path=log, seen=seen)
        ew.record_block(t0, "acct1", "AAPL", "Bollinger Mean Reversion", "buy", 15, path=log, seen=seen)
        ew.record_block(t0, "acct1", "AAPL", "VWAP Mean Reversion", "sell", 15, path=log, seen=seen)
        check("a different account, ticker, strategy or side is a new line (5 in all)", len(log.read_text().splitlines()) == 5)
        ew.record_block(utc(2026, 3, 3, 14, 31), "acct1", "AAPL", "VWAP Mean Reversion", "buy", 15, path=log, seen=seen)
        check("the next day is a new line", len(log.read_text().splitlines()) == 6)
        row = json.loads(log.read_text().splitlines()[0])
        check("the line carries time, New York time, account, ticker, strategy, side and window length",
              row["ny_time"] == "2026-03-02 09:31:00" and row["account_id"] == "acct1" and row["ticker"] == "AAPL"
              and row["strategy_name"] == "VWAP Mean Reversion" and row["side"] == "buy" and row["skip_minutes"] == 15
              and row["timestamp"].startswith("2026-03-02T14:31:00"))
        sell_row = json.loads(log.read_text().splitlines()[4])
        check("a sell is logged as a sell (a short entry)", sell_row["side"] == "sell" and sell_row["strategy_name"] == "VWAP Mean Reversion")
        bad = Path(td) / "no_such_dir" / "x.jsonl"
        seen2: set = set()
        check("a write that fails returns False and raises nothing", ew.record_block(t0, "a", "T", "S", "buy", 15, path=bad, seen=seen2) is False)
        (Path(td) / "no_such_dir").mkdir()
        check("a failed write is retried on the next poll (not marked as seen)", ew.record_block(t0, "a", "T", "S", "buy", 15, path=bad, seen=seen2) is True)
        check("a bad argument returns False and raises nothing", ew.record_block(None, "a", "T", "S", "buy", 15, path=log, seen=set()) is False)
    check("the log lives in the trader's state folder", ew.BLOCKS_PATH.name == "entry_blocks.jsonl" and ew.BLOCKS_PATH.parent.name == "auto_trader_state")
    i_rec = seg.find("entry_window.record_block(")
    check("the trader logs a held-back entry, after the status line and before `continue`",
          i_rec != -1 and seg.find("opening_reason is not None") < i_rec < seg.find("continue", i_rec)
          and seg.find("continue", seg.find("opening_reason is not None")) > i_rec)
    call = seg[i_rec: seg.find(")\n", seg.find("control.skip_opening_minutes", i_rec)) + 1]
    check("the trader passes now, the account, ticker, strategy, side and setting to the logger, in that order",
          " ".join(call.split()) == "entry_window.record_block( datetime.now(timezone.utc), broker_account.account_id, ticker, strategy_name, order_side.value, control.skip_opening_minutes, )",
          " ".join(call.split()))

    # --- the report (entry_block_report.py): synthetic blocks and a synthetic read-only trades database
    import entry_block_report as rep  # noqa: E402
    with tempfile.TemporaryDirectory() as td:
        db = Path(td) / "t.db"
        con = sqlite3.connect(db)
        con.execute("create table live_trades (id integer primary key, account_id text, ticker text, strategy_name text, is_paper int, entry_time text, entry_price real, exit_time text, exit_price real, qty real, pnl real, pnl_pct real, conviction real)")

        def add(acct, tk, strat, paper, entry):
            con.execute("insert into live_trades (account_id, ticker, strategy_name, is_paper, entry_time, pnl_pct) values (?,?,?,?,?,0)", (acct, tk, strat, paper, entry))

        add("a", "AAPL", "VWAP Mean Reversion", 0, "2026-03-02T15:10:00+00:00")      # later the same day: the blocked signal was DELAYED
        add("a", "MSFT", "VWAP Mean Reversion", 0, "2026-03-02T14:40:00+00:00")      # inside the window: a LEAK
        add("a", "GOOG", "VWAP Mean Reversion", 0, "2026-03-02T16:00:00+00:00")      # outside the window, no block: counts as an equity entry only
        add("a", "X:BTCUSD", "VWAP Mean Reversion", 0, "2026-03-02T14:40:00+00:00")  # crypto in the window: not a leak
        add("a", "AMZN", "VWAP Mean Reversion", 1, "2026-03-02T14:40:00+00:00")      # paper: ignored
        add("a", "OLD", "VWAP Mean Reversion", 0, "2026-02-27T14:40:00+00:00")       # before the since date: ignored
        add("b", "AMD", "VWAP Mean Reversion", 0, "2026-03-02T16:10:00+00:00")      # same ticker and strategy, but ANOTHER account
        add("a", "TSLA", "Bollinger Mean Reversion", 0, "2026-03-02T16:20:00+00:00")  # same ticker, but ANOTHER strategy
        add("a", "LATE", "VWAP Mean Reversion", 0, "2026-03-02T03:00:00+00:00")      # 22:00 on 1 March in New York: before the since date there
        con.commit()
        con.close()
        trades = rep.read_trades(db, "2026-03-02")
        blocks = [
            {"timestamp": "2026-03-02T14:31:00+00:00", "ny_time": "2026-03-02 09:31:00", "account_id": "a", "ticker": "AAPL", "strategy_name": "VWAP Mean Reversion", "side": "buy", "skip_minutes": 15},
            {"timestamp": "2026-03-02T14:33:00+00:00", "ny_time": "2026-03-02 09:33:00", "account_id": "a", "ticker": "NVDA", "strategy_name": "VWAP Mean Reversion", "side": "buy", "skip_minutes": 15},
            {"timestamp": "2026-02-27T14:33:00+00:00", "ny_time": "2026-02-27 09:33:00", "account_id": "a", "ticker": "OLD", "strategy_name": "VWAP Mean Reversion", "side": "buy", "skip_minutes": 15},
            {"timestamp": "2026-03-02T14:35:00+00:00", "ny_time": "2026-03-02 09:35:00", "account_id": "a", "ticker": "AMD", "strategy_name": "VWAP Mean Reversion", "side": "buy", "skip_minutes": 15},
            {"timestamp": "2026-03-02T14:36:00+00:00", "ny_time": "2026-03-02 09:36:00", "account_id": "a", "ticker": "TSLA", "strategy_name": "VWAP Mean Reversion", "side": "buy", "skip_minutes": 15},
        ]
        r = rep.build_report(blocks, trades, 15, "2026-03-02")
        check("report: counts only blocks on or after the since date", r["blocked"] == 4 and r["per_day"] == {"2026-03-02": 4})
        check("report: a real-money equity entry in the window is a leak; crypto, paper and earlier entries are not", [t["ticker"] for t in r["leaks"]] == ["MSFT"])
        check("report: equity entries counted are the real-money ones on or after the New York date (AAPL, MSFT, GOOG, AMD, TSLA)", r["equity_entries"] == 5, str(r["equity_entries"]))
        check("report: delayed needs the same ticker, strategy AND account later the same day (only AAPL); the rest are skipped", r["delayed"] == 1 and r["skipped"] == 3, f"{r['delayed']} {r['skipped']}")
        # the exit code: 1 when something leaked, 0 when not (the real paths are swapped for the test only)
        import io
        from contextlib import redirect_stdout
        old_paths = (rep.DB_PATH, ew.BLOCKS_PATH)
        try:
            rep.DB_PATH, ew.BLOCKS_PATH = db, Path(td) / "blocks_for_main.jsonl"
            ew.BLOCKS_PATH.write_text("\n".join(json.dumps(b) for b in blocks) + "\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()) as out:
                code_leak = rep.main(["--since", "2026-03-02"])
            text = out.getvalue()
            check("report: exits 1 and names the leak when an entry slipped through", code_leak == 1 and "LEAK" in text and "MSFT" in text)
            check("report: prints the held-back count and the delayed/skipped split", "Entries held back: 4" in text and "delayed, not skipped): 1" in text and "really skipped): 3" in text, text)
            with redirect_stdout(io.StringIO()):
                code_clean = rep.main(["--since", "2026-03-02", "--skip-minutes", "1"])
            check("report: exits 0 when nothing slipped through (window set to 1 minute)", code_clean == 0)
        finally:
            rep.DB_PATH, ew.BLOCKS_PATH = old_paths
        check("report: a database opened read-only refuses a write", _readonly_refuses(db))
        check("report: opens the database read-only (source says mode=ro)", "mode=ro" in (HERE / "entry_block_report.py").read_text(encoding="utf-8"))
        check("report: a missing database or blocks file gives empty lists, not a crash",
              rep.read_trades(Path(td) / "none.db", "2026-03-02") == [] and rep.read_blocks(Path(td) / "none.jsonl") == [])
        half = Path(td) / "half.jsonl"
        half.write_text(json.dumps(blocks[0]) + "\n" + '{"timestamp": "2026', encoding="utf-8")
        check("report: a half-written last line is skipped", len(rep.read_blocks(half)) == 1)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
