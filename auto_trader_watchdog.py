"""Keeps auto_trader.py running: launches it, then relaunches it if it ever
dies while this watchdog stays alive.

WHY THIS EXISTS (2026-08-24 outage): auto_trader.py's own Startup-folder
shortcut (TradingBotAutoStart, see backtester/CLAUDE_NOTES.txt) only
relaunches it on login - a mid-session crash (the process ran fine, then an
uncaught exception killed it at 21:00 while the user was still logged in)
just left it dead until someone happened to notice, for 108 minutes that
time. auto_trader.py's own main() loop now catches and logs per-cycle
exceptions instead of dying (see the 2026-08-24 comment in main()), which
closes the most likely cause - but this watchdog is the backstop for
whatever that doesn't cover (a crash during startup, before the loop's own
try/except is even active; something OS-level killing the process outright).

Same shape as Mobile_App/backend/watchdog.py, adapted for a process with no
HTTP health endpoint: health is read from auto_trader_state's own status.json
(pid + heartbeat) instead of an HTTP call, checking BOTH the OS-level process
handle and heartbeat freshness for the same reason that file's
_another_instance_alive() does - a hard-killed process's last-written
heartbeat can still look "fresh" for a few minutes, and a hung-but-not-killed
process still holds a live PID, so either signal alone can be fooled.

WHY A LOOP, NOT TASK SCHEDULER - see Mobile_App/backend/watchdog.py's own
doc comment; same reasoning, same no-admin-rights constraint found when
auto_trader.py's own Startup shortcut was first set up.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import logging_setup  # noqa: E402
from backtester.auto_trader_state import load_control, load_status  # noqa: E402

ROOT = Path(__file__).resolve().parent
PYTHONW = ROOT / ".venv" / "Scripts" / "pythonw.exe"
CHECK_INTERVAL_SECONDS = 60
# status.json's heartbeat is only written at the very END of run_cycle - AFTER
# every ticker and every broker round-trip that cycle, not incrementally - so
# there's no signal at all until the whole first cycle finishes. 30s was the
# original value here and it was a real, live mistake (2026-08-24): tonight's
# own logs showed one legitimate full cycle taking ~58s end to end, so a 30s
# grace killed every single cold start before it could ever write its first
# heartbeat, in a tight loop, and - because _another_instance_alive() only
# sees a fresh heartbeat as "someone's already running", not a live-but-slow
# PID - one of those repeated relaunches raced its own predecessor into
# existence and produced two simultaneous auto_trader.py processes, a real
# double-order risk that a 20-minute execution-log check confirmed (by luck,
# not by design) never actually placed a duplicate trade. 10 minutes is
# deliberately generous rather than tuned tight - missing a genuine hang for
# a few extra minutes is a far smaller cost than repeating that mistake.
STARTUP_GRACE_SECONDS = 600


def _is_pid_alive(pid: int) -> bool:
    """Same OS-level check as auto_trader.py's own _is_pid_alive /
    diagnose_bot.py's copy - kept as a third copy rather than imported,
    since this script must stay importable (and startable) even if
    auto_trader.py itself is what's currently broken."""
    if os.name == "nt":
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_healthy() -> bool:
    status = load_status()
    if not status.running or not status.pid or not status.last_heartbeat:
        return False
    if not _is_pid_alive(status.pid):
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
    except ValueError:
        return False
    control = load_control()
    # Same slack (~3 poll cycles, floor 5 min) as auto_trader's own
    # _another_instance_alive - a live loop beats at least once per interval.
    return age < max(300, 3 * control.poll_interval_seconds)


LOCK_PATH = ROOT / "auto_trader_state" / "watchdog.lock"


def _pid_command_line(pid: int) -> str:
    """This pid's command line, or "" if it can't be read."""
    script = (
        f"(Get-CimInstance Win32_Process -Filter 'ProcessId={pid}')"
        ".CommandLine"
    )
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def _other_watchdog_running() -> bool:
    """True if ANOTHER auto_trader_watchdog.py process is already alive.

    THE lesson from 2026-08-24. Two auto_trader.py processes ran
    simultaneously that night - a real double-order risk (no duplicate trade
    was actually placed; the execution log was checked). The cause was NOT a
    race inside one watchdog, as first assumed: it was two WATCHDOGS running
    at once, each supervising its own child, after a botched manual launch
    sequence started a second one before the first had been stopped.
    auto_trader.py's own _another_instance_alive() guard could not save it -
    that checks for a FRESH HEARTBEAT, and a just-launched instance hasn't
    written one yet, so during a cold start two children both look like the
    only one. A supervisor that can be accidentally run twice is a supervisor
    that can double whatever it supervises, so the guard belongs here too.

    Uses a LOCK FILE holding the owning pid, deliberately NOT a scan for
    processes whose command line mentions this script: that scan was tried
    first (2026-08-25) and false-positived immediately, because ANY process
    that merely names the file - the shell command that launched it, a grep,
    an editor - matches too, and a watchdog that refuses to start because
    someone once typed its name is worse than no guard.

    A stale lock (hard kill, power cut) can't wedge it shut: the recorded pid
    must ALSO still be alive AND still be running this script, checked against
    its real command line, before it counts as a live owner.
    """
    try:
        recorded = int(LOCK_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return False  # no lock, or unreadable/garbage - treat as free
    if recorded == os.getpid() or not _is_pid_alive(recorded):
        return False
    # Guards against pid REUSE: the OS may have handed this number to
    # something unrelated since the lock was written.
    return "auto_trader_watchdog.py" in _pid_command_line(recorded)


def _claim_lock() -> None:
    """Record this process as the watchdog owner. Best-effort: failing to
    write the lock must not stop a legitimate watchdog from supervising the
    trader - the guard above simply degrades to 'no lock found'."""
    try:
        LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
        LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill the launched process AND any children.

    proc.pid is the venv's pythonw.exe launcher STUB, which spawns the real
    pythonw3.13.exe as a child. Terminating the stub was measured (2026-08-25)
    to take the child with it, but taskkill /T makes that explicit rather than
    relying on the observed behaviour of a launcher we don't control - the
    consequence of getting it wrong is an orphaned trading process.
    """
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True, timeout=30,
        )
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def _launch() -> subprocess.Popen:
    print("[watchdog] launching auto_trader...")
    return subprocess.Popen(
        [str(PYTHONW), str(ROOT / "auto_trader.py")],
        cwd=str(ROOT),
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


def _kill_pid_tree(pid: int) -> None:
    """Kill a process tree by PID - for an auto_trader this watchdog did NOT
    launch (e.g. started by restart_all.py or the Startup shortcut) and has no
    Popen handle for. Without this, an unhealthy foreign instance would be
    left running while a replacement started alongside it: the exact
    double-instance this file exists to prevent."""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    logging_setup.configure("auto_trader_watchdog")
    if _other_watchdog_running():
        print("[watchdog] another watchdog is already running - exiting to avoid "
              "supervising a second auto_trader (see _other_watchdog_running).")
        return
    _claim_lock()

    # Adopt a healthy existing auto_trader rather than starting a second one.
    # Health is judged from status.json (see _is_healthy), NOT from owning the
    # process handle, so this watchdog can supervise an instance it didn't
    # launch - started by restart_all.py, or by the Startup shortcut. Without
    # this, merely STARTING the watchdog while the bot was already running
    # would itself create the double instance it's meant to prevent.
    proc: subprocess.Popen | None = None
    launched_at = time.monotonic()
    if _is_healthy():
        print("[watchdog] auto_trader already healthy - adopting it, not launching a second.")
    else:
        proc = _launch()
        time.sleep(STARTUP_GRACE_SECONDS)

    while True:
        proc_dead = proc is not None and proc.poll() is not None
        if proc_dead or not _is_healthy():
            reason = "process exited" if proc_dead else "no fresh heartbeat"
            print(f"[watchdog] auto_trader down ({reason}, "
                  f"{round(time.monotonic() - launched_at)}s since launch) - relaunching")
            # An unhealthy instance we DIDN'T launch still has to go before a
            # replacement starts, or the two coexist. status.pid is the real
            # auto_trader pid (it writes os.getpid() itself each cycle).
            if proc is None:
                stale_pid = load_status().pid
                if stale_pid and _is_pid_alive(stale_pid):
                    print(f"[watchdog] killing unhealthy foreign auto_trader pid {stale_pid}")
                    _kill_pid_tree(stale_pid)
                    time.sleep(5)
            if proc is not None and not proc_dead:
                # Kill the whole tree, then WAIT for the OS to confirm it's
                # gone before starting a replacement: kill/taskkill only
                # REQUEST termination, so launching immediately after could
                # briefly overlap a slow-to-die process with its replacement.
                # If it won't die within the timeout, skip relaunching this
                # cycle rather than risk a second instance - the next health
                # check tries again.
                _kill_tree(proc)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("[watchdog] old process wouldn't die within 30s - "
                          "NOT launching a replacement this cycle, will recheck")
                    time.sleep(CHECK_INTERVAL_SECONDS)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            launched_at = time.monotonic()
            proc = _launch()
            time.sleep(STARTUP_GRACE_SECONDS)
        time.sleep(CHECK_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
