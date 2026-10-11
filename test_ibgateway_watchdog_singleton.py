"""Regression test for ibgateway_watchdog.py's singleton guard, which was
shipped on 2026-10-10 and did not work.

Its liveness check was `os.kill(pid, 0)` - the Unix idiom. On Windows signal
0 is CTRL_C_EVENT: the call asks Windows to send Ctrl+C to process GROUP
`pid`, which fails for a windowless pythonw process, and the failure was read
as "that pid is not alive". So the guard concluded, every time, that no other
instance was running. It had been checked in isolation with stand-ins, on
purpose, to avoid starting a real second watchdog next to the real one.

Found 2026-10-11 by doing exactly that: a second instance started watching,
overwrote the state file with its own pid, and had to be killed by hand.

Everything below uses REAL processes and the real functions. Only the state
file's location is redirected, so the live watchdog's own file is untouched.

    .venv/Scripts/python test_ibgateway_watchdog_singleton.py
"""
from __future__ import annotations

import json
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


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -  {detail}" if detail and not ok else ""))


def write_state(pid: int | None, heartbeat: datetime | None) -> None:
    wd._STATE_PATH.write_text(
        json.dumps({"pid": pid, "last_heartbeat": heartbeat.isoformat() if heartbeat else None}), encoding="utf-8"
    )


def main() -> int:
    now = datetime.now(timezone.utc)
    with tempfile.TemporaryDirectory() as tmp:
        wd._STATE_PATH = Path(tmp) / "ibgateway_watchdog_state.json"

        # A real, live, windowless process - the same kind of process the
        # real watchdog is (pythonw has no console; that is what made
        # os.kill(pid, 0) fail and the old check answer "not alive").
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        other = subprocess.Popen([str(pythonw if pythonw.exists() else sys.executable), "-c", "import time; time.sleep(60)"])
        try:
            time.sleep(1)
            print("LIVENESS")
            check("a live process is reported alive", wd._is_pid_alive(other.pid) is True)
            check("the probe did not disturb it", other.poll() is None)
            check("this process is reported alive", wd._is_pid_alive(os.getpid()) is True)

            print("SINGLETON GUARD")
            check("no state file -> free to start", wd._another_instance_alive() is False)

            write_state(other.pid, now)
            check("live pid + fresh heartbeat -> another instance IS running", wd._another_instance_alive() is True,
                  "this is the case the shipped guard got wrong")

            write_state(other.pid, now - timedelta(hours=2))
            check("live pid + stale heartbeat -> not a live watchdog (hung, or the pid was reused)",
                  wd._another_instance_alive() is False)

            write_state(os.getpid(), now)
            check("our own pid -> never blocks ourselves", wd._another_instance_alive() is False)

            wd._STATE_PATH.write_text("{not json", encoding="utf-8")
            check("unreadable state file -> free to start", wd._another_instance_alive() is False)

            write_state(other.pid, now)
        finally:
            other.kill()
            other.wait(timeout=10)

        time.sleep(1)
        check("a dead process is reported dead", wd._is_pid_alive(other.pid) is False)
        check("dead pid + fresh heartbeat -> free to start (a hard-killed watchdog must not block its replacement)",
              wd._another_instance_alive() is False)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED: " + "; ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
