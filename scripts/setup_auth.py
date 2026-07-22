"""One-time (or re-run anytime) setup for the dashboard's password gate.

Run this yourself in a terminal:

    python scripts/setup_auth.py

It prompts for a username and password locally (nothing is sent anywhere),
hashes the password, and writes auth_config.yaml. That file is gitignored —
keep it that way, it's effectively your login credential store.
"""

from __future__ import annotations

import getpass
import secrets
from pathlib import Path

import streamlit_authenticator as stauth
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "auth_config.yaml"


def main() -> None:
    print("Dashboard login setup")
    print("======================")

    existing = {}
    if CONFIG_PATH.exists():
        existing = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        print(f"Found existing config at {CONFIG_PATH} — this will update it.")

    username = input("Username [admin]: ").strip() or "admin"
    name = input("Display name [Admin]: ").strip() or "Admin"

    while True:
        password = getpass.getpass("Password: ")
        if len(password) < 8:
            print("Password must be at least 8 characters. Try again.")
            continue
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords didn't match. Try again.")
            continue
        break

    hashed = stauth.Hasher.hash(password)
    del password, confirm  # don't linger in memory longer than needed

    credentials = existing.get("credentials", {"usernames": {}})
    credentials.setdefault("usernames", {})
    credentials["usernames"][username] = {
        "email": existing.get("credentials", {})
        .get("usernames", {})
        .get(username, {})
        .get("email", f"{username}@localhost"),
        "name": name,
        "password": hashed,
        "logged_in": False,
        "failed_login_attempts": 0,
    }

    cookie_key = existing.get("cookie", {}).get("key") or secrets.token_hex(32)

    config = {
        "credentials": credentials,
        "cookie": {
            "name": "backtester_dashboard_auth",
            "key": cookie_key,
            "expiry_days": 7,
        },
    }

    CONFIG_PATH.write_text(yaml.safe_dump(config, default_flow_style=False))
    print(f"\nSaved to {CONFIG_PATH}. Start the dashboard with: streamlit run app.py")


if __name__ == "__main__":
    main()
