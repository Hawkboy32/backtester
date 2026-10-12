"""Regression test for ibgateway_watchdog.py starting / restarting IB Gateway
(added 2026-10-12).

Until then the watchdog only notified. Gateway was found dead for 25.5 hours
over the weekend of 2026-10-10 and, in the IBC logs, sitting on its login
screen for 22 hours the day before - nothing on the machine restarts it.

Each recovery attempt is a login, and a login sends a 2FA push to the user's
phone, so what matters most here is when it must NOT act. This walks every
branch of that decision, then exercises the parts that touch the machine with
real processes: the real process query (read-only), a real launch of a
stand-in StartGateway.bat, and a real stop of a stand-in Gateway.

It never starts or stops the real Gateway: _try_recovery is only ever called
with the process query, the stop and the start replaced by stand-ins.

    .venv/Scripts/python test_ibgateway_watchdog_recovery.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ibgateway_watchdog as wd  # noqa: E402 - importing runs no watchdog loop (main() is guarded)

PASS, FAIL = [], []
NO_WINDOW = subprocess.CREATE_NO_WINDOW


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -  {detail}" if detail and not ok else ""))


def local_time(hour: int) -> datetime:
    """A timezone-aware moment at `hour` o'clock local time."""
    return datetime.now().astimezone().replace(hour=hour, minute=30, second=0, microsecond=0)


def decide(**overrides):
    args = dict(
        now=local_time(12), outage_minutes=45.0, gateway=wd._GatewayProcesses(),
        attempts=[], paused=False, is_online=lambda: True,
    )
    args.update(overrides)
    return wd._recovery_action(**args)


def pids_matching(needle: str) -> list[int]:
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -like '*{needle}*' -and $_.Name -ne 'powershell.exe' }} | ForEach-Object {{ $_.ProcessId }}"],
        capture_output=True, text=True, timeout=30, creationflags=NO_WINDOW,
    ).stdout
    return [int(x) for x in out.split() if x.isdigit()]


def main() -> int:
    live = wd._GatewayProcesses(live=[111], wrappers=[100])
    noon = local_time(12)

    print("PARSING THE PROCESS QUERY")
    parsed = wd._parse_gateway_processes("wrapper 1516\nlive 6680\ndone\n")
    check("live Gateway and its console are both found", parsed is not None and parsed.live == [6680] and parsed.wrappers == [1516] and not parsed.other)
    parsed = wd._parse_gateway_processes("done\n")
    check("nothing running is an empty result, not a failure", parsed is not None and not parsed.live and not parsed.wrappers and not parsed.other)
    parsed = wd._parse_gateway_processes("other 4242\ndone\n")
    check("a paper Gateway / TWS is reported as someone else holding the login", parsed is not None and parsed.other)
    check("output without the closing marker is a failed query", wd._parse_gateway_processes("live 6680\n") is None)
    check("empty output is a failed query", wd._parse_gateway_processes("") is None)
    check("unrecognised output is a failed query", wd._parse_gateway_processes("Get-CimInstance : Access denied\ndone\n") is None)

    print("WHEN IT ACTS")
    check("no process, past the grace period -> start", decide(outage_minutes=wd.GRACE_MINUTES) == ("start", ""))
    check("alive but unreachable for STUCK_MINUTES -> restart", decide(gateway=live, outage_minutes=wd.STUCK_MINUTES) == ("restart", ""))
    check("only a leftover console, no Gateway -> start", decide(gateway=wd._GatewayProcesses(wrappers=[100]))[0] == "start")

    print("WHEN IT MUST NOT ACT")
    check("no process but still inside the grace period", decide(outage_minutes=wd.GRACE_MINUTES - 1) == (None, ""))
    check("alive and unreachable for less than STUCK_MINUTES (may still be logging in)", decide(gateway=live, outage_minutes=wd.STUCK_MINUTES - 1) == (None, ""))
    action, why = decide(gateway=None)
    check("the process query failed", action is None and "could not tell" in why, why)
    action, why = decide(gateway=wd._GatewayProcesses(other=True))
    check("a paper Gateway or TWS is logged in", action is None and "paper" in why, why)
    action, why = decide(gateway=wd._GatewayProcesses(live=[111], other=True))
    check("... even with a live Gateway process also present", action is None and "paper" in why, why)
    action, why = decide(paused=True)
    check("the pause file exists", action is None and "paused" in why, why)
    for hour in (23, 0, 3, 6):
        action, why = decide(now=local_time(hour))
        check(f"quiet hours: {hour:02d}:30 local", action is None and "between" in why, why)
    for hour in (7, 12, 22):
        check(f"not quiet hours: {hour:02d}:30 local", decide(now=local_time(hour))[0] == "start")
    action, why = decide(attempts=None)
    check("its own attempt record is unreadable", action is None and "unreadable" in why, why)
    three = [noon - timedelta(hours=h) for h in (20, 10, 2)]
    action, why = decide(attempts=three)
    check(f"{wd.RECOVERY_MAX_PER_24H} attempts already made in 24 hours", action is None and "already tried 3" in why, why)
    action, why = decide(attempts=[noon - timedelta(minutes=wd.RECOVERY_MIN_SPACING_MINUTES - 1)])
    check("the last attempt was under an hour ago", action is None and "last tried" in why, why)
    check("the last attempt was just over an hour ago -> acts", decide(attempts=[noon - timedelta(minutes=wd.RECOVERY_MIN_SPACING_MINUTES + 1)])[0] == "start")
    check("two attempts, both over an hour old -> acts", decide(attempts=[noon - timedelta(hours=5), noon - timedelta(hours=2)])[0] == "start")
    action, why = decide(is_online=lambda: False)
    check("no internet connection", action is None and "internet" in why, why)
    asked = []
    decide(outage_minutes=1, is_online=lambda: asked.append(1) or True)
    check("the internet probe is not made when nothing is due", not asked)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wd._RECOVERY_PATH = tmp_path / "ibgateway_watchdog_recovery.json"
        wd._PAUSE_PATH = tmp_path / "ibgateway_watchdog.pause"
        now = datetime.now(timezone.utc)

        print("THE ATTEMPT RECORD")
        check("no file yet = no attempts", wd._load_recovery_attempts(now) == [])
        ok = wd._record_recovery_attempt([now - timedelta(hours=30), now - timedelta(hours=3)], now)
        loaded = wd._load_recovery_attempts(now)
        check("attempts are written and read back", ok and loaded is not None and len(loaded) == 2, str(loaded))
        check("attempts older than 24 hours no longer count", loaded is not None and all(now - s < timedelta(hours=24) for s in loaded))
        wd._RECOVERY_PATH.write_text("{not json", encoding="utf-8")
        check("a corrupt record reads as None (limit reached), not as empty", wd._load_recovery_attempts(now) is None)
        wd._RECOVERY_PATH.unlink()

        print("THE WHOLE STEP, with the machine-touching parts replaced")
        calls: list[str] = []
        sent: list[tuple] = []
        state = {"gateway": wd._GatewayProcesses(live=[111], wrappers=[100])}
        real = (wd._gateway_processes, wd._stop_gateway, wd._start_gateway, wd._is_online, wd.notifications.notify, wd._in_quiet_hours)
        wd._gateway_processes = lambda: state["gateway"]
        wd._stop_gateway = lambda g: calls.append("stop") or True
        wd._start_gateway = lambda: calls.append("start")
        wd._is_online = lambda: True
        wd._in_quiet_hours = lambda hour: False  # this block must give the same answer whenever the test is run
        wd.notifications.notify = lambda *a, **k: sent.append((a, k)) or True
        try:
            acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 45.0)
            check("stuck Gateway: stopped, then started, in that order", acted and calls == ["stop", "start"], f"{acted} {why} {calls}")
            check("... and counted", len(wd._load_recovery_attempts(datetime.now(timezone.utc)) or []) == 1)
            check("... and the user is told to approve the login", len(sent) == 1 and "Approve" in sent[0][0][1] and "1 of 3" in sent[0][0][1], str(sent))
            calls.clear()
            acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 47.0)
            check("asked again two minutes later: does nothing", not acted and not calls and "last tried" in why, f"{acted} {why} {calls}")

            wd._RECOVERY_PATH.unlink()
            state["gateway"] = wd._GatewayProcesses()
            acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 12.0)
            check("no Gateway at all: started without a stop", acted and calls == ["start"], f"{acted} {why} {calls}")

            calls.clear()
            wd._RECOVERY_PATH.unlink()
            wd._stop_gateway = lambda g: calls.append("stop") and False
            state["gateway"] = wd._GatewayProcesses(live=[111])
            acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 45.0)
            check("the old Gateway will not stop: a second one is NOT started", not acted and calls == ["stop"] and "could not be stopped" in why, f"{acted} {why} {calls}")
            check("... and that still used up an attempt", len(wd._load_recovery_attempts(datetime.now(timezone.utc)) or []) == 1)

            calls.clear()
            wd._RECOVERY_PATH.unlink()
            state["gateway"] = wd._GatewayProcesses()
            real_write = wd.atomic_write_text

            def disk_full(*_a, **_k):
                raise OSError("disk full")

            wd.atomic_write_text = disk_full
            try:
                acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 45.0)
            finally:
                wd.atomic_write_text = real_write
            check("the attempt cannot be recorded: nothing is started", not acted and not calls and "could not write" in why, f"{acted} {why} {calls}")

            calls.clear()
            acted, why = wd._try_recovery("IBKR Live", "192.168.1.50", 45.0)
            check("Gateway on another machine: nothing is started", not acted and not calls and "another machine" in why, why)

            wd._PAUSE_PATH.write_text("", encoding="utf-8")
            acted, why = wd._try_recovery("IBKR Live", "127.0.0.1", 45.0)
            check("pause file present: nothing is started", not acted and not calls and "paused" in why, why)
            wd._PAUSE_PATH.unlink()
        finally:
            wd._gateway_processes, wd._stop_gateway, wd._start_gateway, wd._is_online, wd.notifications.notify, wd._in_quiet_hours = real

        print("THE REAL PROCESS QUERY (read-only)")
        found = wd._gateway_processes()
        check("the query runs on this machine and is understood", found is not None)
        if found is not None:
            print(f"        right now: live Gateway pids {found.live}, IBC consoles {found.wrappers}, other session: {found.other}")

        print("A REAL LAUNCH, of a stand-in StartGateway.bat")
        bat = tmp_path / "StartGateway.bat"
        marker = f"wdrecovery{os.getpid()}"
        # The same shape as the real one: write nothing to the caller, `start` a minimized console, return.
        bat.write_text(
            "@echo off\r\n"
            'set > "%~dp0env.txt"\r\n'
            f'start "{marker}" /Min cmd /c "ping -n 41 127.0.0.1 > nul & rem {marker}"\r\n'
            "exit /B\r\n",
            encoding="ascii",
        )
        real_bat = wd.START_GATEWAY_BAT
        wd.START_GATEWAY_BAT = bat
        try:
            wd._start_gateway()
            deadline = time.monotonic() + 15
            started: list[int] = []
            while time.monotonic() < deadline and not started:
                time.sleep(1)
                started = pids_matching(marker)
            check("the window it starts is running", bool(started), "no process found")
            env_text = (tmp_path / "env.txt").read_text(encoding="utf-8", errors="replace") if (tmp_path / "env.txt").exists() else ""
            check("it was started in the script's own folder with an environment", "SystemRoot=" in env_text or "SYSTEMROOT=" in env_text.upper())
            from_dotenv = sorted(k for k in os.environ if k not in wd._LOGIN_ENV)
            leaked = [k for k in from_dotenv if f"\n{k}=" in "\n" + env_text]
            check(f"none of the {len(from_dotenv)} variables loaded from .env reach it", not leaked, f"{len(leaked)} leaked")
        finally:
            wd.START_GATEWAY_BAT = real_bat
            for pid in pids_matching(marker):
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)

        print("A REAL STOP, of a stand-in Gateway")
        # A console process with a child under it, like the IBC console and its java.
        stand_in = subprocess.Popen(
            ["cmd.exe", "/c", f"ping -n 43 127.0.0.1 > nul & rem {marker}stop"], creationflags=NO_WINDOW,
        )
        time.sleep(2)
        before = pids_matching("-n 43 127.0.0.1")  # the console (its command line holds the text) and ping itself
        real_query = wd._gateway_processes
        wd._gateway_processes = lambda: wd._GatewayProcesses(wrappers=[stand_in.pid] if stand_in.poll() is None else [])
        try:
            stopped = wd._stop_gateway(wd._GatewayProcesses(wrappers=[stand_in.pid]))
        finally:
            wd._gateway_processes = real_query
        time.sleep(1)
        after = pids_matching("-n 43 127.0.0.1")
        check("the stand-in console and the process under it were running", len(before) >= 2, str(before))
        check("stop reports success and the whole tree is gone", stopped and not after, f"stopped={stopped} left={after}")
        if stand_in.poll() is None:
            subprocess.run(["taskkill", "/PID", str(stand_in.pid), "/T", "/F"], capture_output=True, creationflags=NO_WINDOW)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed" + (": " + "; ".join(FAIL) if FAIL else ""))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
