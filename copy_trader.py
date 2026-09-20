"""Watch an eToro Popular Investor's live portfolio and print BUY/SELL-shaped
events as they happen — the "test on the go" observation tool for the
copy-trading idea (2026-09-16). Personal system only, not part of
ChopperCommercial.

Deliberately OBSERVE-ONLY: this never places an order, on eToro or anywhere
else. It exists to answer "what would copying this investor actually have
looked like" before any execution wiring is worth building at all — there's
no way to backtest a copy-trading signal (eToro doesn't expose per-position
history), so watching it live is the only way to evaluate it.

Run in a terminal (foreground, Ctrl+C to stop) — not managed by
restart_all.py or any watchdog yet; this is a manual observation tool, not a
production background process.

Usage:
    python copy_trader.py --username Linareswillian
    python copy_trader.py --username Aukie2008 --username ca_sual --username campervans \
        --username celesh --username RainbirdFx --interval 90

Watching several investors in one process (not several separate ones) keeps
polling coordinated against eToro's 60-requests/60s live-portfolio rate
limit — N usernames means N requests per cycle, spread across one sleep
loop, rather than N independently-timed processes that could all land in
the same second.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import keyring

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.accounts import KEYRING_SERVICE, _keyring_key, _load_raw  # noqa: E402
from backtester.copy_trading import CopyTradingError, poll_investor  # noqa: E402

# eToro's live-portfolio endpoint is rate-limited to 60 requests/60s — this
# default leaves a wide margin for a single watched investor.
DEFAULT_INTERVAL_SECONDS = 90


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _find_etoro_credentials(account_nickname: str | None) -> tuple[str, str, str]:
    """Reuse whichever eToro account is already linked via the dashboard's
    Accounts tab, rather than asking for keys a second time. Returns
    (nickname, api_key, user_key)."""
    candidates = [a for a in _load_raw() if a["broker"] == "etoro"]
    if account_nickname:
        candidates = [a for a in candidates if a["nickname"] == account_nickname]
    if not candidates:
        raise SystemExit(
            "No linked eToro account found (Accounts tab). Link one first, or pass "
            "--account-nickname if you have more than one."
        )
    if len(candidates) > 1:
        names = ", ".join(a["nickname"] for a in candidates)
        raise SystemExit(f"Multiple eToro accounts linked ({names}) — pass --account-nickname to pick one.")

    account = candidates[0]
    api_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "api_key"))
    user_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "secret_key"))
    if not api_key or not user_key:
        raise SystemExit(f"Credentials for '{account['nickname']}' are missing from the OS keyring.")
    return account["nickname"], api_key, user_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--username", required=True, action="append",
        help="eToro username to watch — repeat for multiple (e.g. --username A --username B)",
    )
    parser.add_argument("--account-nickname", default=None, help="Which linked eToro account's keys to use")
    parser.add_argument(
        "--interval", type=int, default=DEFAULT_INTERVAL_SECONDS, help=f"Poll interval in seconds (default {DEFAULT_INTERVAL_SECONDS})"
    )
    parser.add_argument("--once", action="store_true", help="Poll once and exit, instead of looping")
    args = parser.parse_args()

    # Python buffers stdout more aggressively once it's not attached to a
    # real terminal (piped to a file/log, or a background-task runner) — this
    # is a long-running loop whose whole point is to be watched/tailed
    # between polls, so every print() must actually reach the file promptly
    # rather than sitting in a buffer until it fills or the process exits.
    sys.stdout.reconfigure(line_buffering=True)

    account_nickname, api_key, user_key = _find_etoro_credentials(args.account_nickname)
    usernames = args.username
    print(f"[{_now()}] Watching {len(usernames)} investor(s) using eToro account '{account_nickname}': "
          f"{', '.join(usernames)}. Ctrl+C to stop.")

    while True:
        for username in usernames:
            try:
                events = poll_investor(username, api_key, user_key)
            except CopyTradingError as e:
                print(f"[{_now()}] {username}: poll failed: {e}")
            except Exception as e:  # noqa: BLE001 — a transient network hiccup on
                # ONE investor (confirmed live: a plain requests.ReadTimeout,
                # not wrapped in CopyTradingError since it never got a response
                # to check) must never kill observation of the other four, or
                # the whole process — same "one bad item can't take down the
                # loop" principle scan_runner.py already follows.
                print(f"[{_now()}] {username}: poll failed (unexpected: {type(e).__name__}: {e})")
            else:
                if not events:
                    print(f"[{_now()}] {username}: no change")
                for event in events:
                    print(f"[{_now()}] {event.describe()}")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
