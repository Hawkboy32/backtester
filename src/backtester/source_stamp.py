"""Detects a long-running process still executing OLD code after the source
on disk has changed.

WHY THIS EXISTS (two real outages, 2026-08-12 and 2026-08-13): every
long-running process here — auto_trader.py, the Streamlit dashboard, the
mobile backend — imports its modules once at startup and then holds them in
memory indefinitely. Editing a file changes NOTHING for a process already
running. Both outages were the same shape: a field was added to a persisted
dataclass, one process was restarted, the others weren't, and the ones still
running couldn't read what the restarted one wrote.

auto_trader_state.from_dict now stops that from CRASHING anything, but it
can't stop the subtler half: an old process quietly using the DEFAULT for a
field it can't see, and showing you a number that isn't what's stored. Only
"this process is running stale code" catches that — which is what this does.

Deliberately hashes source only, never state files: the question is "does
the running code match the code on disk", not "has data changed".
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json

STAMPS_PATH = STATE_DIR / "process_stamps.json"

# Everything whose change should invalidate a running process. src/backtester
# covers the shared library all three processes import; the two top-level
# scripts are their own entry points.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WATCHED = [
    _PROJECT_ROOT / "src" / "backtester",
    _PROJECT_ROOT / "auto_trader.py",
    _PROJECT_ROOT / "app.py",
]


def compute_source_hash() -> str:
    """One hash over every watched .py file's path + contents. Sorted so it's
    order-stable, and content-based rather than mtime-based so a touch or a
    checkout that rewrites timestamps doesn't produce a false 'stale'."""
    digest = hashlib.sha256()
    for base in _WATCHED:
        if base.is_file():
            paths = [base]
        elif base.is_dir():
            paths = sorted(base.rglob("*.py"))
        else:
            continue
        for path in paths:
            if "__pycache__" in path.parts:
                continue
            digest.update(str(path.relative_to(_PROJECT_ROOT)).encode("utf-8"))
            try:
                digest.update(path.read_bytes())
            except OSError:
                continue  # a file vanishing mid-walk shouldn't break the check
    return digest.hexdigest()[:16]


def record_start(process_name: str) -> None:
    """Call once at startup. Records the source hash this process actually
    loaded, so a later comparison against disk is real evidence rather than a
    guess. Never raises — a diagnostic must not be able to stop a trading
    process from booting."""
    try:
        stamps = read_state_json(STAMPS_PATH, default={})
    except Exception:  # noqa: BLE001 — a corrupt stamp file is not worth blocking startup
        stamps = {}
    stamps[process_name] = {
        "source_hash": compute_source_hash(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    try:
        atomic_write_text(STAMPS_PATH, json.dumps(stamps, indent=2))
    except Exception:  # noqa: BLE001
        pass


# How to recognise each long-running process from its command line. Shared
# with restart_all.py so the two can never disagree about what's supposed to
# be running.
PROCESS_MATCHERS = {
    "dashboard": "streamlit",
    "mobile_backend": "signal_api",
    "auto_trader": "auto_trader.py",
}


def running_pids(match: str) -> list[int]:
    """PIDs whose command line contains `match`. PowerShell because tasklist
    alone doesn't expose command lines on Windows."""
    import subprocess
    script = (
        "Get-CimInstance Win32_Process | "
        f"Where-Object {{ $_.CommandLine -like '*{match}*' }} | "
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
    for token in out.split():
        try:
            pid = int(token.strip())
        except ValueError:
            continue
        if pid:
            pids.append(pid)
    return pids


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return str(pid) in out
        except Exception:  # noqa: BLE001
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stale_processes() -> list[dict]:
    """Processes running code that isn't the code on disk.

    Driven by what's ACTUALLY RUNNING (PROCESS_MATCHERS), not by what happens
    to be in the stamp file. That distinction matters: an earlier version of
    this walked the stamps and skipped any entry whose PID was dead, which
    silently reported "all clear" while an unstamped process was running —
    caught 2026-08-13 when the watchdog relaunched the mobile backend behind
    restart_all's back, leaving an orphaned stamp and a live process nobody
    had verified. A dead stamp means UNKNOWN, not fine; treating unknown as
    fine is the same mistake as the silent-empty state loads.

    Each result has status:
      "stale"      — running, stamped, hash differs from disk (restart it)
      "unverified" — running, but no live stamp matches it, so what code it
                     loaded cannot be confirmed (restart it to be sure)
    A process that simply isn't running is omitted; stopped isn't stale.
    """
    try:
        stamps = read_state_json(STAMPS_PATH, default={})
    except Exception:  # noqa: BLE001
        stamps = {}
    current = compute_source_hash()
    out: list[dict] = []

    for name, match in PROCESS_MATCHERS.items():
        pids = running_pids(match)
        if not pids:
            continue  # not running at all — nothing to be stale about

        info = stamps.get(name) if isinstance(stamps.get(name), dict) else None
        stamped_pid = info.get("pid") if info else None
        stamp_is_live = (
            info is not None
            and isinstance(stamped_pid, int)
            and stamped_pid in pids
            and _pid_alive(stamped_pid)
        )

        if stamp_is_live and info.get("source_hash") == current:
            continue  # verified current

        out.append({
            "process": name,
            "pid": stamped_pid if stamp_is_live else pids[0],
            "pids": pids,
            "started_at": info.get("started_at") if info else None,
            "started_hash": info.get("source_hash") if info else None,
            "current_hash": current,
            "status": "stale" if stamp_is_live else "unverified",
        })
    return out
