"""Wait for the US session to close, then run restart_all.py.

Exists so the trader is restarted at a moment when it cannot be mid-decision on
an open position, without anyone having to sit and watch the clock. Polls the
broker's own market clock rather than trusting a computed close time — a half
day (Thanksgiving, Christmas Eve) closes early and a hardcoded 21:00 UTC would
restart into a live session.

Safety: it only ever RESTARTS. It never changes configuration, never places or
closes an order, and if the clock can't be read it waits rather than acting.
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from backtester.accounts import build_broker_accounts, list_accounts  # noqa: E402

POLL_SECONDS = 60
# Enough past the bell that any in-flight closing order has settled, but short
# enough that the restart still happens promptly after the close.
GRACE_SECONDS = 120
MAX_WAIT_HOURS = 8


def market_is_open() -> bool | None:
    """True/False, or None if no account could tell us (treated as 'keep waiting')."""
    for account in list_accounts():
        if account.get("broker") != "alpaca":
            continue
        try:
            clock = build_broker_accounts([account["id"]])[0].get_market_clock()
        except Exception:  # noqa: BLE001
            continue
        if clock is not None:
            return bool(clock["is_open"])
    return None


def main() -> int:
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    print(f"[{datetime.now(timezone.utc):%H:%M}] waiting for the session to close...", flush=True)

    while time.time() < deadline:
        state = market_is_open()
        if state is False:
            print(f"[{datetime.now(timezone.utc):%H:%M}] market closed — "
                  f"waiting {GRACE_SECONDS}s for in-flight orders to settle", flush=True)
            time.sleep(GRACE_SECONDS)
            # Re-check: never restart into a session that reopened (or a clock
            # blip) while we were sleeping.
            if market_is_open() is not False:
                print("market no longer reports closed — resuming wait", flush=True)
                continue
            break
        if state is None:
            print(f"[{datetime.now(timezone.utc):%H:%M}] clock unreadable — waiting", flush=True)
        time.sleep(POLL_SECONDS)
    else:
        print(f"gave up after {MAX_WAIT_HOURS}h without seeing a close — NOT restarting", flush=True)
        return 1

    print(f"[{datetime.now(timezone.utc):%H:%M}] running restart_all.py", flush=True)
    result = subprocess.run(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), str(ROOT / "restart_all.py")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=600,
    )
    print(result.stdout, flush=True)
    if result.stderr.strip():
        print("stderr:", result.stderr, flush=True)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
