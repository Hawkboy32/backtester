"""Local store for linked brokerage accounts.

Credentials are encrypted at rest via the `keyring` library, which delegates
to the OS's native secure credential store (Windows Credential Manager /
macOS Keychain / Linux Secret Service) — protected by your OS login, not
a separate password this app manages. broker_accounts.json (gitignored)
holds only non-secret metadata: nickname, broker type, paper/live mode,
and a last-4-characters hint for display. Credentials are only ever
written by the user themselves via the dashboard's own masked input
fields, never typed, seen, or handled by anything else.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import keyring
import keyring.errors

from backtester.brokers.alpaca import AlpacaBroker
from backtester.brokers.base import BrokerAccount
from backtester.brokers.coinbase import CoinbaseBroker
from backtester.brokers.kraken import KrakenBroker
from backtester.brokers.tastytrade_broker import TastytradeBroker

ACCOUNTS_PATH = Path(__file__).resolve().parent.parent.parent / "broker_accounts.json"
KEYRING_SERVICE = "backtester-dashboard"

# cred_fields are (label for the value stored as "api_key", label for the value
# stored as "secret_key") — the underlying storage is always two secret
# strings regardless of what the broker actually calls them.
BROKER_META: dict[str, dict] = {
    "alpaca": {
        "label": "Alpaca",
        "cred_fields": ("API key", "Secret key"),
        "supports_paper": True,
        "has_market_hours": True,  # US equities — regular session; has an open/closed clock
    },
    "coinbase": {
        "label": "Coinbase (Advanced Trade)",
        "cred_fields": ("API key", "API secret"),
        "supports_paper": False,  # no verified sandbox — always routes to the real live API
        "has_market_hours": False,  # crypto trades 24/7 — always "open"
    },
    "kraken": {
        "label": "Kraken",
        "cred_fields": ("API key", "Private key (API secret)"),
        "supports_paper": False,  # no verified sandbox — always routes to the real live API
        "has_market_hours": False,  # crypto trades 24/7 — always "open"
    },
    "tastytrade": {
        "label": "Tastytrade",
        "cred_fields": ("Refresh token", "Client secret (provider secret)"),
        "supports_paper": True,
        "has_market_hours": True,  # US equities/options — regular session
    },
}
SUPPORTED_BROKERS = list(BROKER_META.keys())


def _load_raw() -> list[dict]:
    if not ACCOUNTS_PATH.exists():
        return []
    return json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))


def _save_raw(accounts: list[dict]) -> None:
    ACCOUNTS_PATH.write_text(json.dumps(accounts, indent=2), encoding="utf-8")


def _keyring_key(account_id: str, field: str) -> str:
    return f"{account_id}:{field}"


def list_accounts() -> list[dict]:
    """Return account metadata — safe to render in the UI, no secrets included."""
    accounts = _load_raw()
    return [
        {
            "id": a["id"],
            "nickname": a["nickname"],
            "broker": a["broker"],
            "is_paper": a["is_paper"],
            "api_key_masked": f"****{a['api_key_last4']}" if a.get("api_key_last4") else "(not set)",
        }
        for a in accounts
    ]


def add_account(nickname: str, broker: str, is_paper: bool, api_key: str, secret_key: str) -> str:
    if broker not in SUPPORTED_BROKERS:
        raise ValueError(f"Unsupported broker '{broker}'. Supported: {SUPPORTED_BROKERS}")

    if not BROKER_META[broker]["supports_paper"]:
        is_paper = False  # no verified sandbox for this broker — never claim paper safety we can't back up

    accounts = _load_raw()
    account_id = str(uuid.uuid4())

    keyring.set_password(KEYRING_SERVICE, _keyring_key(account_id, "api_key"), api_key)
    keyring.set_password(KEYRING_SERVICE, _keyring_key(account_id, "secret_key"), secret_key)

    accounts.append(
        {
            "id": account_id,
            "nickname": nickname,
            "broker": broker,
            "is_paper": is_paper,
            "api_key_last4": api_key[-4:] if len(api_key) >= 4 else "*" * len(api_key),
        }
    )
    _save_raw(accounts)
    return account_id


def remove_account(account_id: str) -> None:
    for field in ("api_key", "secret_key"):
        try:
            keyring.delete_password(KEYRING_SERVICE, _keyring_key(account_id, field))
        except keyring.errors.PasswordDeleteError:
            pass  # already gone / never set — fine, we're removing it anyway

    accounts = [a for a in _load_raw() if a["id"] != account_id]
    _save_raw(accounts)


def build_broker_accounts(account_ids: list[str] | None = None) -> list[BrokerAccount]:
    """Instantiate live BrokerAccount objects for the given account ids
    (or all linked accounts if account_ids is None).
    """
    accounts = _load_raw()
    if account_ids is not None:
        accounts = [a for a in accounts if a["id"] in account_ids]

    built: list[BrokerAccount] = []
    for a in accounts:
        api_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(a["id"], "api_key"))
        secret_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(a["id"], "secret_key"))
        if not api_key or not secret_key:
            raise ValueError(
                f"Credentials for account '{a['nickname']}' are missing from the OS keyring "
                "(deleted outside this app, or a different OS user account). Remove and re-link it."
            )

        broker = a["broker"]
        if broker == "alpaca":
            obj = AlpacaBroker(nickname=a["nickname"], api_key=api_key, secret_key=secret_key, is_paper=a["is_paper"])
        elif broker == "coinbase":
            obj = CoinbaseBroker(nickname=a["nickname"], api_key=api_key, api_secret=secret_key)
        elif broker == "kraken":
            obj = KrakenBroker(nickname=a["nickname"], api_key=api_key, api_secret=secret_key)
        elif broker == "tastytrade":
            obj = TastytradeBroker(
                nickname=a["nickname"],
                refresh_token=api_key,
                provider_secret=secret_key,
                is_paper=a["is_paper"],
            )
        else:
            raise ValueError(f"Unsupported broker '{broker}' for account {a['nickname']}")

        obj.account_id = a["id"]
        built.append(obj)
    return built
