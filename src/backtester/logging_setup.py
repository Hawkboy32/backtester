"""Makes a pythonw.exe-launched process's output actually go somewhere.

WHY THIS EXISTS (2026-08-24 outage): auto_trader.py died silently for ~108
minutes with no trace anywhere - no crash log, no Windows Error Reporting
entry, nothing. It's launched via pythonw.exe (no console window, by design -
see CLAUDE_NOTES.txt), which means stdout/stderr have nowhere to go; every
print() and every uncaught traceback vanishes into the void the moment it's
written. The mobile backend's own watchdog.py has the exact same blind spot
in its own print() calls.

This redirects stdout/stderr to a rotating log file AND installs a
sys.excepthook, so both routine prints and a genuinely uncaught exception
(the kind that would otherwise kill the process with zero evidence) land
somewhere a human can actually read afterward.

Deliberately NOT the `logging` module's own basicConfig-to-file approach:
these scripts are full of plain print() calls, and rewriting every one of
them into logger.info(...) calls is a much bigger, riskier change than
capturing the stream those prints already go through.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_MAX_BYTES = 2_000_000  # ~2MB per file
_BACKUP_COUNT = 3  # ~8MB total per process, plenty for "what happened before it died"


class _StreamToFile:
    """Minimal stdout/stderr replacement - just enough for print() and
    traceback.print_exception() to work, timestamped per line so a log can be
    correlated against other systems (heartbeat.json, current_signals.json)
    after the fact."""

    def __init__(self, handler: RotatingFileHandler):
        self._handler = handler
        self._buffer = ""

    def write(self, text: str) -> None:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line:
                ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                self._handler.stream.write(f"[{ts}] {line}\n")
        self._handler.flush()

    def flush(self) -> None:
        self._handler.flush()


def configure(process_name: str) -> Path:
    """Call once, as early as possible in a script's startup - before
    anything that could itself raise. Returns the log file path.

    Idempotent-ish in practice: each call opens its own handler, so calling
    it twice would duplicate output, but every entry point here only calls
    it once at the top of main()."""
    log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{process_name}.log"

    handler = RotatingFileHandler(
        log_path, maxBytes=_MAX_BYTES, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    stream = _StreamToFile(handler)
    sys.stdout = stream
    sys.stderr = stream

    def _excepthook(exc_type, exc_value, exc_tb) -> None:
        # Last-resort net: anything that escapes every try/except in the
        # script's own code still gets written here before the process dies,
        # instead of disappearing the way the 2026-08-24 outage did.
        print("UNCAUGHT EXCEPTION - process is about to exit:")
        for line in traceback.format_exception(exc_type, exc_value, exc_tb):
            print(line.rstrip("\n"))

    sys.excepthook = _excepthook
    print(f"=== {process_name} logging started (pid={__import__('os').getpid()}) ===")
    return log_path
