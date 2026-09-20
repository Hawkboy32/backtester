"""Local storage for API keys, backed by a gitignored .env file.

Keys never leave this machine: the dashboard writes them straight to .env
via python-dotenv and reads them back into the process environment. Values
are masked wherever they'd otherwise be displayed.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv, set_key

ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

KNOWN_KEYS = {
    "POLYGON_API_KEY": "Polygon.io",
    "OANDA_API_KEY": "OANDA (forex live data)",
    "ANTHROPIC_API_KEY": "Anthropic (Claude advisor — mobile app, added 2026-09-18)",
}


def ensure_env_file() -> None:
    if not ENV_PATH.exists():
        ENV_PATH.touch()


def save_key(name: str, value: str) -> None:
    ensure_env_file()
    set_key(str(ENV_PATH), name, value)
    load_dotenv(ENV_PATH, override=True)


def get_key(name: str) -> str | None:
    load_dotenv(ENV_PATH, override=True)
    return os.environ.get(name)


def list_keys() -> dict[str, str | None]:
    ensure_env_file()
    values = dotenv_values(ENV_PATH)
    return {name: values.get(name) for name in KNOWN_KEYS}


def mask(value: str | None) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]
