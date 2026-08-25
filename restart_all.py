"""Restart every long-running process that reads shared state, in one command.

WHY: editing a file changes nothing for a process already running — each one
imported its modules at startup and holds them in memory. Restarting only
SOME of them is what caused both 2026-08-12's roster wipe and 2026-08-13's
"bot looks stopped" outage: a field was added, one process picked it up, the
others didn't, and they disagreed about what the state files meant.

Three processes share the backtester library and its state files:
  - auto_trader.py    (the trading loop)
  - app.py            (the Streamlit dashboard)
  - Mobile_App/backend/signal_api.py  (the phone's API)

Order matters: the trader goes LAST so it spends the least time down. Its
absence is the only one with a real cost — the other two are read/UI layers.

Usage:
    python restart_all.py                              # restart everything
    python restart_all.py --check                      # report what's stale, change nothing
    python restart_all.py --only dashboard,mobile_backend   # restart just these

--only exists for the case where the market is OPEN with live positions and a
change is additive enough not to affect the trading loop: the two read/UI
layers can be brought up to date immediately while auto_trader keeps running
until the close. Use it deliberately — anything that changes what the trader
does, or what a shared state file MEANS, must restart everything together, or
you recreate exactly the disagreement this script exists to prevent.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import source_stamp  # noqa: E402

ROOT = Path(__file__).resolve().parent
MOBILE_BACKEND = ROOT.parent / "Mobile_App" / "backend"

# (label, match-substring, working dir, launch argv or None)
#
# mobile_backend has argv=None DELIBERATELY: watchdog.py already owns that
# process and relaunches it within ~30s of it going unhealthy. Starting it
# here too raced the watchdog and produced two competing instances fighting
# over port 8600 — one died, leaving an orphaned stamp and an unverified
# survivor (2026-08-13). Stopping it and letting its own supervisor bring it
# back is both simpler and the watchdog's actual job.
PROCESSES = [
    (
        "dashboard",
        "streamlit",
        ROOT,
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "streamlit", "run", "app.py",
         "--server.headless", "true"],
    ),
    (
        "mobile_backend",
        "signal_api",
        MOBILE_BACKEND,
        None,  # watchdog relaunches it — see note above
    ),
    (
        "auto_trader",
        "auto_trader.py",
        ROOT,
        # argv=None once auto_trader_watchdog.py owns this process (2026-08-25),
        # for exactly the same reason mobile_backend has None above: the
        # watchdog relaunches it within ~60s of it going unhealthy, so starting
        # it here too would race the watchdog and risk the two-instance
        # double-order scenario that watchdog exists to prevent. Stopping it and
        # letting its supervisor bring it back is the supervisor's actual job.
        # NOTE: the watchdog's own STARTUP_GRACE_SECONDS means a restart here
        # takes up to ~60s to be noticed, longer than the old direct launch.
        None,
    ),
]


def _find_pids(match: str) -> list[int]:
    """PIDs whose command line contains `match`. Uses WMIC-equivalent via
    PowerShell because tasklist alone doesn't expose command lines."""
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -like '*" + match + "*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:  # noqa: BLE001
        return []
    pids = []
    for line in out.split():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid != 0:
            pids.append(pid)
    return pids


def _stop(pids: list[int]) -> None:
    for pid in pids:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force -ErrorAction SilentlyContinue"],
            capture_output=True, timeout=30,
        )


def check() -> int:
    stale = source_stamp.stale_processes()
    if not stale:
        print("All running processes match the current source. Nothing to restart.")
        return 0
    print("These are NOT running the code currently on disk:")
    for s in stale:
        if s["status"] == "stale":
            print(f"  - {s['process']:16s} STALE       pid {s['pid']:<8} started {s['started_at']}")
        else:
            print(f"  - {s['process']:16s} UNVERIFIED  pids {s['pids']} — running, but no matching "
                  "stamp, so its code can't be confirmed")
    print("\nRun `python restart_all.py` to bring them up to date.")
    return 1


def restart_all(only: set[str] | None = None) -> int:
    print(f"Current source hash: {source_stamp.compute_source_hash()}\n")
    if only:
        unknown = only - {p[0] for p in PROCESSES}
        if unknown:
            print(f"Unknown process name(s): {sorted(unknown)}")
            print(f"Valid names: {sorted(p[0] for p in PROCESSES)}")
            return 2
        print(f"Restarting ONLY: {', '.join(sorted(only))}")
        print("Everything else keeps running its current code.\n")
    for label, match, cwd, argv in PROCESSES:
        if only and label not in only:
            print(f"{label:16s} skipped (--only)")
            continue
        pids = _find_pids(match)
        # Never let this script kill itself (it matches nothing above, but be
        # explicit rather than relying on that).
        pids = [p for p in pids if p != __import__("os").getpid()]
        if pids:
            print(f"{label:16s} stopping {pids}")
            _stop(pids)
            time.sleep(2)
        else:
            print(f"{label:16s} (not running)")

        if argv is None:
            print(f"{label:16s} leaving relaunch to its watchdog (~30-60s)")
            continue

        exe = Path(argv[0])
        if not exe.exists():
            print(f"{label:16s} SKIPPED — interpreter not found at {exe}")
            continue
        subprocess.Popen(
            argv, cwd=str(cwd),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
        )
        print(f"{label:16s} started")
        time.sleep(3)

    # Long enough to cover the SLOWEST watchdog-owned relaunch, since two
    # processes now come back on their supervisor's schedule rather than ours:
    # the mobile backend's watchdog polls every 30s (+10s startup grace), and
    # auto_trader's polls every 60s (2026-08-25). 90s covers the 60s worst-case
    # detection plus the few seconds a relaunched process needs to record its
    # source stamp - at 50s this reported a perfectly healthy auto_trader as
    # STILL STALE simply because its watchdog hadn't noticed yet.
    print("\nWaiting for processes to come up and record their stamps (~90s)...")
    time.sleep(90)
    remaining = source_stamp.stale_processes()
    # Anything deliberately skipped is still stale BY DESIGN — report it as a
    # standing reminder rather than a failure, so it can't be quietly forgotten.
    skipped = [s for s in remaining if only and s["process"] not in only]
    unexpected = [s for s in remaining if not only or s["process"] in only]
    if skipped:
        print("Deliberately left on older code (--only), restart when convenient:")
        for s in skipped:
            print(f"  - {s['process']} pid {s.get('pid')}")
    if unexpected:
        print("STILL STALE (may just need a few more seconds, re-run --check):")
        for s in unexpected:
            print(f"  - {s['process']} pid {s.get('pid')}")
        return 1
    target = "Selected processes" if only else "All processes"
    print(f"{target} restarted and matching current source.")
    return 0


def _parse_only(argv: list[str]) -> set[str] | None:
    for i, arg in enumerate(argv):
        if arg.startswith("--only="):
            return {s.strip() for s in arg.split("=", 1)[1].split(",") if s.strip()}
        if arg == "--only" and i + 1 < len(argv):
            return {s.strip() for s in argv[i + 1].split(",") if s.strip()}
    return None


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else restart_all(_parse_only(sys.argv[1:])))
