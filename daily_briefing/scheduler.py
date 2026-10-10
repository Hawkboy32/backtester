"""Standalone sleep loop with date-stamp gating - the same shape
auto_trader.py's own `_maybe_check_roster_gap` already uses, the only
"run something once a day" pattern proven on this machine (no Task
Scheduler without admin rights, same reason auto_trader_watchdog.py /
ibgateway_watchdog.py are loops too, not scheduled tasks). Runs the real
graph once per day after a cutoff hour and delivers the result via
notifications.notify_daily_briefing - never touches a broker-mutating
call itself (see graph.py / activity_node.py / risk_node.py).

Its own `state.json` (last_run_date), not auto_trader_state's - this is a
separate process writing on its own schedule; sharing a state file with
auto_trader.py would be a write race between two independent processes.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester import notifications  # noqa: E402

from graph import run_briefing  # noqa: E402

STATE_PATH = Path(__file__).resolve().parent / "state.json"
CHECK_INTERVAL_SECONDS = 300  # how often the loop wakes to check the cutoff - not how often it runs
CUTOFF_HOUR_UTC = 21  # ~4-5pm ET, after the US market closes


def _load_last_run_date() -> str | None:
    if not STATE_PATH.exists():
        return None
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8")).get("last_run_date")
    except Exception:  # noqa: BLE001 - an unreadable state file should fail safe (re-run today), not crash
        return None


def _save_last_run_date(date_str: str) -> None:
    STATE_PATH.write_text(json.dumps({"last_run_date": date_str}), encoding="utf-8")


def _should_run_now(now: datetime) -> bool:
    today = now.strftime("%Y-%m-%d")
    if _load_last_run_date() == today:
        return False
    return now.hour >= CUTOFF_HOUR_UTC


def run_forever(*, provider: str = "ollama", model: str | None = None) -> None:
    print(
        f"[daily_briefing] scheduler started - runs once/day after {CUTOFF_HOUR_UTC}:00 UTC, "
        f"checking every {CHECK_INTERVAL_SECONDS}s",
        flush=True,
    )
    while True:
        now = datetime.now(timezone.utc)
        if _should_run_now(now):
            today = now.strftime("%Y-%m-%d")
            print(f"[daily_briefing] running for {today}...", flush=True)
            try:
                result = run_briefing(provider=provider, model=model)
                briefing = result.get("briefing", "")
                notifications.notify_daily_briefing(briefing)
                print(f"[daily_briefing] sent:\n{briefing}", flush=True)
            except Exception as exc:  # noqa: BLE001 - a failed run must not crash the loop; try again tomorrow
                print(f"[daily_briefing] run failed: {exc}", flush=True)
            # Saved even on failure - same "gate on attempted, not just
            # succeeded" reasoning as _maybe_check_roster_gap, so a broken
            # run doesn't retry every 5 minutes for the rest of the day.
            _save_last_run_date(today)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model")
    args = parser.parse_args()
    run_forever(provider=args.provider, model=args.model)
