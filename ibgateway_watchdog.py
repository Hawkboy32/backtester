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

SINCE 2026-10-12 IT ALSO STARTS / RESTARTS GATEWAY, not only notifies. Until
then nothing did: IBC handles the daily 03:00 restart of a Gateway that is
running and logged in, and that is all. The IBC logs back to August showed
two ways it stayed down for hours with only a notification to show for it:
  - the Gateway process ended (a 2FA push nobody approved, after which IBC is
    configured to exit; or its console window being closed) - five times,
    the last for 25.5 hours over a weekend;
  - Gateway alive but sitting on its login screen after unanswered 2FA
    prompts - 22 hours on 2026-10-09/10, a full trading day.
So: unreachable past the grace period with no Gateway process -> start it;
alive but unreachable for STUCK_MINUTES -> stop it and start it again.

EVERY ATTEMPT IS A LOGIN, AND A LOGIN IS A 2FA PUSH TO THE USER'S PHONE. That
is what the limits in _recovery_action() are for: one attempt an hour, three
in any 24 hours, none overnight, none while a paper Gateway or TWS is logged
in on the same login (the user is doing that on purpose), none when it cannot
tell what is running. When it will not act it says why in the notification.
To switch the whole thing off without touching code, create the file
ibgateway_watchdog.pause next to this script; delete it to switch back on.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

# Taken BEFORE .env is loaded: this is the environment Gateway is started
# with, and it has no business inheriting every broker key in that file.
_LOGIN_ENV = dict(os.environ)

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

# Recovery (see the module docstring). Each attempt sends the user a 2FA push.
STUCK_MINUTES = 30  # alive but unreachable this long = not going to log itself in; GRACE_MINUTES applies when there is no process at all
RECOVERY_MIN_SPACING_MINUTES = 60
RECOVERY_MAX_PER_24H = 3
QUIET_HOURS = (23, 7)  # local time, [start, end): nobody approves a push at 03:00, and it would burn the day's attempts
START_GATEWAY_BAT = Path(
    os.environ.get("IBGATEWAY_START_BAT")
    or Path(__file__).resolve().parent.parent / "ibc" / "IBC" / "StartGateway.bat"
)  # the same script the IBGatewayAutoStart login shortcut runs
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}

_STATE_PATH = Path(__file__).resolve().parent / "ibgateway_watchdog_state.json"
# Separate from _STATE_PATH on purpose: that file is the heartbeat Sentrywatch
# reads and is rewritten whole every cycle; this one has to survive the
# watchdog itself being restarted, or every restart would reset the limits.
_RECOVERY_PATH = Path(__file__).resolve().parent / "ibgateway_watchdog_recovery.json"
_PAUSE_PATH = Path(__file__).resolve().parent / "ibgateway_watchdog.pause"


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
    """Real OS-level liveness check - the same tasklist approach as
    auto_trader.py / auto_trader_watchdog.py, and for a reason found the hard
    way on 2026-10-11.

    The first version of this function (2026-10-10) was the Unix idiom
    `os.kill(pid, 0)` on its own. On Windows that is NOT a liveness probe:
    signal 0 is CTRL_C_EVENT, so it asks Windows to send Ctrl+C to process
    GROUP `pid`. For a windowless pythonw process that call fails, the
    OSError was read as "not alive", and the singleton guard below never
    fired for anyone - it was shipped, documented as fixed, and did nothing.
    Found only when a second instance was deliberately started against a
    running one as a live test: it started watching, overwrote the state
    file with its own pid, and had to be killed by hand. The guard had been
    "verified" in isolation with stand-ins precisely to avoid running that
    live test. test_ibgateway_watchdog_singleton.py now runs it against real
    processes."""
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return str(pid) in out.stdout
        except Exception:  # noqa: BLE001
            return False
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


# ---------------------------------------------------------------------------
# Recovery: starting / restarting Gateway. See the module docstring for why
# and for what the limits protect.
# ---------------------------------------------------------------------------

@dataclass
class _GatewayProcesses:
    live: list[int] = field(default_factory=list)      # IBC-run Gateway in live mode (java.exe)
    wrappers: list[int] = field(default_factory=list)  # the `cmd /K` console StartGateway.bat opens; outlives its java
    other: bool = False  # a paper Gateway, TWS, or a Gateway not started through IBC - someone else has the login


# Classifies in PowerShell and returns only "kind pid" lines: Gateway's command
# line can carry credentials, so it never leaves that process to be logged here.
# The closing "done" tells "nothing is running" apart from "the query failed".
_GATEWAY_QUERY = r"""
Get-CimInstance Win32_Process -Filter "Name='java.exe' OR Name='cmd.exe' OR Name='ibgateway.exe' OR Name='tws.exe'" | ForEach-Object {
  $c = [string]$_.CommandLine
  $kind = $null
  if ($_.Name -eq 'cmd.exe') { if ($c -like '*\scripts\DisplayBannerAndLaunch.bat*') { $kind = 'wrapper' } }
  elseif ($_.Name -ne 'java.exe') { $kind = 'other' }
  elseif ($c -like '*ibcalpha.ibc.IbcGateway*' -and $c.TrimEnd() -match '\slive$') { $kind = 'live' }
  elseif ($c -like '*ibcalpha.ibc.*') { $kind = 'other' }
  if ($kind) { '{0} {1}' -f $kind, $_.ProcessId }
}
'done'
"""


def _parse_gateway_processes(output: str) -> _GatewayProcesses | None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or lines[-1] != "done":
        return None
    found = _GatewayProcesses()
    for line in lines[:-1]:
        kind, _, pid = line.partition(" ")
        if kind == "other":
            found.other = True
        elif kind in ("live", "wrapper") and pid.isdigit():
            (found.live if kind == "live" else found.wrappers).append(int(pid))
        else:
            return None  # output this function does not understand is not "nothing running"
    return found


def _gateway_processes() -> _GatewayProcesses | None:
    """What is running right now, or None if that could not be established -
    in which case nothing is started: a second Gateway on the same login
    throws the first one off."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", _GATEWAY_QUERY],
            capture_output=True, text=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:  # noqa: BLE001
        return None
    return _parse_gateway_processes(out.stdout) if out.returncode == 0 else None


def _load_recovery_attempts(now: datetime) -> list[datetime] | None:
    """Attempts made in the last 24 hours. None if the record exists but
    cannot be read - treated as "limit reached", never as "no attempts yet"."""
    if not _RECOVERY_PATH.exists():
        return []
    try:
        stamps = [datetime.fromisoformat(s) for s in json.loads(_RECOVERY_PATH.read_text(encoding="utf-8"))["attempts"]]
    except Exception:  # noqa: BLE001
        return None
    return [s for s in stamps if now - s < timedelta(hours=24)]


def _record_recovery_attempt(attempts: list[datetime], now: datetime) -> bool:
    try:
        atomic_write_text(_RECOVERY_PATH, json.dumps({"attempts": [s.isoformat() for s in [*attempts, now]]}, indent=2))
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_online() -> bool:
    """Starting Gateway with no internet only produces a failed login."""
    try:
        socket.create_connection(("www.interactivebrokers.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


def _in_quiet_hours(local_hour: int) -> bool:
    start, end = QUIET_HOURS
    return start <= local_hour < end if start <= end else local_hour >= start or local_hour < end


def _recovery_action(
    *,
    now: datetime,
    outage_minutes: float,
    gateway: _GatewayProcesses | None,
    attempts: list[datetime] | None,
    paused: bool,
    is_online: Callable[[], bool],
) -> tuple[str | None, str]:
    """Decide whether to act on an unreachable Gateway: ("start" | "restart" |
    None, why not). The reason is empty when it is simply not time yet. Pure,
    so test_ibgateway_watchdog_recovery.py can walk every branch; `now` is
    timezone-aware."""
    if gateway is None:
        return None, "could not tell what is running"
    if gateway.other:
        return None, "a paper Gateway, TWS or a manually started Gateway is running"
    action, due_after = ("restart", STUCK_MINUTES) if gateway.live else ("start", GRACE_MINUTES)
    if outage_minutes < due_after:
        return None, ""
    if paused:
        return None, f"automatic recovery is paused ({_PAUSE_PATH.name} exists)"
    if _in_quiet_hours(now.astimezone().hour):
        return None, f"it is between {QUIET_HOURS[0]:02d}:00 and {QUIET_HOURS[1]:02d}:00"
    if attempts is None:
        return None, f"{_RECOVERY_PATH.name} is unreadable"
    if len(attempts) >= RECOVERY_MAX_PER_24H:
        return None, f"already tried {len(attempts)} times in the last 24 hours"
    if attempts and (now - max(attempts)) < timedelta(minutes=RECOVERY_MIN_SPACING_MINUTES):
        return None, f"last tried {(now - max(attempts)).total_seconds() / 60:.0f}m ago"
    if not is_online():
        return None, "this PC has no internet connection"
    return action, ""


def _stop_gateway(gateway: _GatewayProcesses) -> bool:
    """Close the IBC console(s) and the Gateway under them. Killing java alone
    is not enough: its console is `cmd /K`, which stays open afterwards, and
    while an autorestart token exists its script starts Gateway again by
    itself. True once no live Gateway is left."""
    for pid in gateway.wrappers:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    for pid in gateway.live:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        remaining = _gateway_processes()
        if remaining is not None and not remaining.live and not remaining.wrappers:
            return True
        time.sleep(2)
    return False


def _start_gateway() -> None:
    """Run StartGateway.bat the way the login shortcut does. It `start`s its
    own minimized console and returns, so Gateway is not this process's child
    and is not in its job object: checked with stand-ins on 2026-10-12, a
    window started this way survives `taskkill /T /F` on the watchdog, which
    is what Sentrywatch does when it replaces it."""
    subprocess.Popen(
        ["cmd.exe", "/c", str(START_GATEWAY_BAT)],
        cwd=str(START_GATEWAY_BAT.parent), env=_LOGIN_ENV, creationflags=subprocess.CREATE_NO_WINDOW,
    )


def _try_recovery(nickname: str, host: str, outage_minutes: float) -> tuple[bool, str]:
    """(acted, why not). When it acts it sends its own notification."""
    if host not in _LOCAL_HOSTS:
        return False, "Gateway runs on another machine"
    if not START_GATEWAY_BAT.exists():
        return False, f"{START_GATEWAY_BAT} does not exist"
    now = datetime.now(timezone.utc)
    gateway = _gateway_processes()
    attempts = _load_recovery_attempts(now)
    action, reason = _recovery_action(
        now=now, outage_minutes=outage_minutes, gateway=gateway, attempts=attempts,
        paused=_PAUSE_PATH.exists(), is_online=_is_online,
    )
    if action is None:
        return False, reason
    # Counted before anything is started, and nothing is started if it cannot
    # be counted: an attempt that is not recorded is an attempt without a limit.
    if not _record_recovery_attempt(attempts, now):
        return False, f"could not write {_RECOVERY_PATH.name}"
    attempt_no = len(attempts) + 1
    was = "was running but not logged in" if action == "restart" else "was not running"
    print(f"[ibgateway_watchdog] {nickname} unreachable for {outage_minutes:.0f}m, Gateway {was} - {action} (attempt {attempt_no}/{RECOVERY_MAX_PER_24H})", flush=True)
    # Also on a plain start: consoles left behind by sessions that already ended.
    if (gateway.live or gateway.wrappers) and not _stop_gateway(gateway):
        print("[ibgateway_watchdog] could not stop the existing Gateway - not starting another", flush=True)
        return False, "the existing Gateway could not be stopped"
    try:
        _start_gateway()
    except Exception as exc:  # noqa: BLE001
        print(f"[ibgateway_watchdog] starting Gateway failed: {exc!r}", flush=True)
        return False, f"starting Gateway failed ({type(exc).__name__})"
    notifications.notify(
        "IB Gateway restarted - approve the login",
        f"{nickname} was unreachable for {outage_minutes:.0f}m and Gateway {was}, so it has been started. "
        f"Approve the IBKR prompt on your phone now. Automatic attempt {attempt_no} of {RECOVERY_MAX_PER_24H} in 24h.",
        priority="high",
    )
    return True, ""


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
            acted, why_not = _try_recovery(nickname, host, outage_min)
            if acted:
                last_notified[nickname] = now  # it sent its own notification
                continue
            already_notified_recently = nickname in last_notified and (now - last_notified[nickname]) / 60 < RENOTIFY_MINUTES
            if already_notified_recently:
                continue
            print(f"[ibgateway_watchdog] {nickname} unreachable for {outage_min:.0f}m - notifying ({why_not or 'not due for a restart yet'})", flush=True)
            notifications.notify(
                "IB Gateway unreachable",
                f"{nickname} has been unreachable for {outage_min:.0f}m ({host}:{port}) - "
                f"likely stuck past the daily restart (check for a pending 2FA approval)."
                + (f" Not restarting it automatically: {why_not}." if why_not else ""),
                priority="high",
            )
            last_notified[nickname] = now

        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
