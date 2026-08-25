"""Human-readable release version for the trading bot system - auto_trader.py,
the dashboard, and the mobile backend (via its own venv, which has this
package installed too) all read the same file.

Separate from source_stamp.py's content hash, which answers "is this process
running the code currently on disk" - this answers "what release is this",
for humans to reference in conversation or a bug report. Bump the VERSION
file by hand alongside a meaningful batch of changes; this module only reads
it.
"""

from __future__ import annotations

from pathlib import Path

_VERSION_PATH = Path(__file__).resolve().parent.parent.parent / "VERSION"
_FALLBACK = "0.0.0-unknown"


def read_version() -> str:
    try:
        return _VERSION_PATH.read_text(encoding="utf-8").strip() or _FALLBACK
    except OSError:
        return _FALLBACK


VERSION = read_version()
