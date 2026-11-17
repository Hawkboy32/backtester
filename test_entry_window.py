"""The opening-minutes rule (added 2026-11-17 from the strategy lab's Study I-H): no NEW equity entry in the first skip_opening_minutes after 09:30 New York.

Opening only: a close, crypto, forex (and so the IG index Opening Spike Fade) are never held back.

    .venv/Scripts/python test_entry_window.py
"""
from __future__ import annotations

import ast
import sys
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
          "opening_reason is not None" in seg and seg[seg.find("opening_reason is not None"):].split("continue", 1)[0].count("\n") < 6)
    app_src = (HERE / "app.py").read_text(encoding="utf-8")
    check("the dashboard's Save carries the setting forward (so saving does not reset it)", "skip_opening_minutes=control.skip_opening_minutes" in app_src)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for f in FAIL:
        print("  FAILED:", f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
