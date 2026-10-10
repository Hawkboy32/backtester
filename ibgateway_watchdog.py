"""Watches IB Gateway's own reachability and notifies (via notifications.py,
the same ntfy-based channel auto_trader.py already uses for trade/risk
alerts) if it's been unreachable longer than expected.

WHY THIS EXISTS. IB Gateway forces a full logout roughly every 24h (an
IBKR-imposed security restart, not a bug) — diagnose_bot.py already surfaces
this after the fact ("IBKR Live: account risk check failed... connection
refused") but only when someone happens to run a pre-flight check. Pairs
with IBC (IB Controller, see docs/ibc_setup.md) which handles the scheduled
restart + auto-relogin — this watchdog's real job is catching the case IBC
can't fully automate: if the account has 2FA enabled, IBKR still wants a
human to approve a push notification, and IBC's own restart can silently
stall waiting on that. Without this, nobody would know until the next
manual pre-flight check or a real trade attempt failed.

WHY A GRACE PERIOD, NOT AN IMMEDIATE ALERT. Gateway's own daily restart is
EXPECTED to have a short reconnect gap even when everything works perfectly
(IBC's relaunch + relogin genuinely takes a couple of minutes) - alerting
on every restart would just be noise the first real failure gets lost in.
GRACE_MINUTES is deliberately longer than that normal cycle.

WHY POLL check_gateway_reachable(), not just trust "the process is running".
Same lesson already learned building watchdog.py (mobile backend) and
auto_trader.py's own singleton guard: a process can be alive but stuck (e.g.
Gateway up but sitting on a login prompt waiting for 2FA) - a live API
connect attempt is the only check that can't be fooled by that.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import accounts as accounts_module  # noqa: E402
from backtester import logging_setup  # noqa: E402
from backtester import notifications  # noqa: E402
from backtester.auto_trader_state import atomic_write_text  # noqa: E402
from backtester.brokers.ibkr import check_gateway_reachable  # noqa: E402

CHECK_INTERVAL_SECONDS = 120
GRACE_MINUTES = 10  # longer than IBC's normal restart+relogin cycle - see module docstring
RENOTIFY_MINUTES = 60  # don't re-alert every single check once already down, just periodically

_STATE_PATH = Path(__file__).resolve().parent / "ibgateway_watchdog_state.json"


@dataclass
class _WatchdogStatus:
    pid: int | None = None
    last_heartbeat: str | None = None


def _load_watchdog_status() -> _WatchdogStatus:
    if not _STATE_PATH.exists():
        return _WatchdogStatus()
    try:
        return _WatchdogStatus(**json.loads(_STATE_PATH.read_text(encoding="utf-8")))
    except Exception:
        return _WatchdogStatus()


def _save_watchdog_status(status: _WatchdogStatus) -> None:
    atomic_write_text(_STATE_PATH, json.dumps(asdict(status), indent=2))


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _another_instance_alive() -> bool:
    """Dual-signal singleton guard, same shape as auto_trader.py's own
    `_another_instance_alive()` (and the same reason: a heartbeat alone can
    still look "fresh" for minutes after a hard kill, and a PID alone can't
    tell a live loop from a hung one - both checks closes both gaps).

    Added 2026-10-10 after this script was found to have NO such guard at
    all, unlike auto_trader.py - a stray manual run started a second, fully
    independent watchdog instance instead of exiting, caught and killed
    within seconds only because it was being watched for directly. Low
    financial stakes compared to auto_trader.py (this script only notifies,
    never trades), but a duplicate watchdog double-notifying - or someday
    double-launching Gateway itself - is exactly the failure mode this
    project's own "adopt, don't duplicate" principle exists to prevent."""
    status = _load_watchdog_status()
    if not status.last_heartbeat or not status.pid or status.pid == os.getpid():
        return False
    if not _is_pid_alive(status.pid):
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
    except ValueError:
        return False
    # a live loop beats once per CHECK_INTERVAL_SECONDS; allow ~3 cycles of slack
    return age < max(300, 3 * CHECK_INTERVAL_SECONDS)


def _touch_heartbeat() -> None:
    _save_watchdog_status(_WatchdogStatus(pid=os.getpid(), last_heartbeat=datetime.now(timezone.utc).isoformat()))


def _ibkr_targets() -> list[tuple[str, str, int]]:
    """(nickname, host, port) for every linked, non-paper IBKR account -
    IBKR Live only as of 2026-08-20, written generically but deliberately
    excludes IBKR Paper even now that it's an active copy-trade target
    (see copy_trade_executor.py's _target_paper_accounts docstring):
    confirmed live 2026-09-21 that IBKR won't keep a live + paper Gateway
    session logged in concurrently under one login, so IBKR Paper is
    expected to be down/unreachable whenever Live (the thing that actually
    matters) is up - watching it here would just be permanent, correct-per-
    design noise, not a real signal. Skipped (not an error) if none are
    linked."""
    targets = []
    for acct in accounts_module.list_accounts():
        if acct["broker"] != "ibkr" or acct["is_paper"]:
            continue
        params = acct.get("conn_params", {})
        host, port = params.get("host"), params.get("port")
        if host and port:
            targets.append((acct["nickname"], host, int(port)))
    return targets


def main() -> None:
    # Launched via pythonw.exe, so every print() below went nowhere until
    # 2026-08-26. Found during a preflight check after IB Gateway sat down
    # for ~5 hours overnight (its 03:00 IBC auto-restart didn't complete):
    # this was the one process that could still fail invisibly, and its log
    # was exactly what would have said whether it noticed. See logging_setup.
    logging_setup.configure("ibgateway_watchdog")
    if _another_instance_alive():
        print(
            "[ibgateway_watchdog] another instance appears to be running (fresh heartbeat) - "
            "exiting to avoid a duplicate watchdog.", flush=True,
        )
        return
    _touch_heartbeat()
    targets = _ibkr_targets()
    if not targets:
        print("[ibgateway_watchdog] no linked live IBKR accounts found - nothing to watch, exiting", flush=True)
        return
    print(f"[ibgateway_watchdog] watching: {[t[0] for t in targets]}", flush=True)

    down_since: dict[str, float] = {}
    last_notified: dict[str, float] = {}

    while True:
        now = time.time()
        _touch_heartbeat()
        for nickname, host, port in targets:
            reachable = check_gateway_reachable(host, port)
            if reachable:
                if nickname in down_since:
                    outage_min = (now - down_since[nickname]) / 60
                    print(f"[ibgateway_watchdog] {nickname} reachable again after {outage_min:.1f}m", flush=True)
                    notifications.notify(
                        "IB Gateway back up", f"{nickname} reconnected after {outage_min:.0f}m unreachable.",
                    )
                down_since.pop(nickname, None)
                last_notified.pop(nickname, None)
                continue

            down_since.setdefault(nickname, now)
            outage_min = (now - down_since[nickname]) / 60
            if outage_min < GRACE_MINUTES:
                continue  # still within IBC's normal restart+relogin window - not an alert yet
            already_notified_recently = nickname in last_notified and (now - last_notified[nickname]) / 60 < RENOTIFY_MINUTES
            if already_notified_recently:
                continue
            print(f"[ibgateway_watchdog] {nickname} unreachable for {outage_min:.0f}m - notifying", flush=True)
            notifications.notify(
                "IB Gateway unreachable",
                f"{nickname} has been unreachable for {outage_min:.0f}m ({host}:{port}) - "
                f"likely stuck past the daily restart (check for a pending 2FA approval).",
                priority="high",
            )
            last_notified[nickname] = now

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
