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
from backtester.brokers.etoro import EToroBroker
from backtester.brokers.ibkr import ASSET_CLASSES as IBKR_ASSET_CLASSES
from backtester.brokers.ibkr import IBKRBroker
from backtester.brokers.ig import IGBroker
from backtester.brokers.kraken import KrakenBroker
from backtester.brokers.oanda import OandaBroker
from backtester.brokers.tastytrade_broker import TastytradeBroker
from backtester.auto_trader_state import atomic_write_text

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
        "asset_classes": frozenset({"equity"}),
    },
    "coinbase": {
        "label": "Coinbase (Advanced Trade)",
        "cred_fields": ("API key", "API secret"),
        "supports_paper": False,  # no verified sandbox — always routes to the real live API
        "has_market_hours": False,  # crypto trades 24/7 — always "open"
        "asset_classes": frozenset({"crypto"}),
    },
    "kraken": {
        "label": "Kraken",
        "cred_fields": ("API key", "Private key (API secret)"),
        "supports_paper": False,  # no verified sandbox — always routes to the real live API
        "has_market_hours": False,  # crypto trades 24/7 — always "open"
        "asset_classes": frozenset({"crypto"}),
    },
    "tastytrade": {
        "label": "Tastytrade",
        "cred_fields": ("Refresh token", "Client secret (provider secret)"),
        "supports_paper": True,
        "has_market_hours": True,  # US equities/options — regular session
        "asset_classes": frozenset({"equity"}),
    },
    "ibkr": {
        "label": "Interactive Brokers",
        "cred_fields": ("", ""),  # unused — IBKR has no API key/secret
        "supports_paper": True,
        "has_market_hours": True,  # US equities — regular session clock
        # Connects to a locally-running IB Gateway/TWS the user logs into. There are
        # no secrets to store; instead we keep non-secret connection config (host,
        # port, clientId, account code) as plain metadata. The dashboard renders a
        # different link form for these, and build_broker_accounts skips the keyring.
        "uses_gateway": True,
        # What the BROKER TYPE can support (both equity and forex sub-accounts).
        # A given LINKED account is still only ever one or the other — see
        # account_asset_class() below, which is the per-account authority this
        # broker-level set does NOT answer on its own.
        "asset_classes": frozenset({"equity", "forex"}),
    },
    "ig": {
        "label": "IG (Phase 3 — forex CFDs, live-verified 2026-07-31/08-01)",
        "cred_fields": ("API key", "Password"),
        # IG's session auth needs THREE credentials (username, password, api_key),
        # not the usual two — this is a deliberate extension of the keyring
        # scheme (see add_account/build_broker_accounts) rather than forcing
        # IG's username into a field labeled "secret key" or similar.
        "extra_cred_field": "Username",
        "supports_paper": True,  # IG has a real, documented demo account (verified 2026-07-21/27)
        "has_market_hours": True,  # forex CFDs — closed weekends, unlike crypto's real 24/7
        "asset_classes": frozenset({"forex"}),
    },
    "oanda": {
        "label": "OANDA (forex — paper execution, added 2026-08-08)",
        # Two credentials, neither of them a traditional "secret key" — an
        # OANDA API token (Bearer auth) plus the account ID every order/
        # position endpoint is scoped to. Reuses the standard 2-field keyring
        # slots rather than needing IG's extra_cred_field mechanism.
        "cred_fields": ("API token", "Account ID"),
        "supports_paper": True,  # OANDA's practice environment — the ONLY thing this is scoped to for now
        "has_market_hours": True,  # spot forex — closed weekends
        "asset_classes": frozenset({"forex"}),
    },
    "etoro": {
        "label": "eToro (added 2026-09-16 — read-only exploration for a planned copy-trading signal)",
        # Public API Key (x-api-key) + User Key (x-user-key), both from
        # Settings -> Trading -> API Key Management on eToro's own site.
        # eToro issues a SEPARATE key per environment (Demo/Real aren't a
        # runtime flag on one key) — linking here as Paper vs Live stores
        # whichever pair the user generated for that environment, same as
        # every other broker's is_paper split.
        "cred_fields": ("Public API Key", "User Key"),
        "supports_paper": True,  # eToro's own Demo environment — a real, documented sandbox
        "has_market_hours": False,  # eToro spans multiple asset classes with different hours;
        # not accurately a single yes/no yet — see asset_classes note below.
        # A single placeholder, not eToro's real range (stocks/crypto/commodities/forex/indices) —
        # account_asset_class() picks exactly one class per non-IBKR account, and nothing routes
        # an eToro account through it yet (no auto-trading target, no scan universe). Revisit once
        # the copy-trading signal work actually needs real per-position asset-class routing.
        "asset_classes": frozenset({"equity"}),
    },
}
SUPPORTED_BROKERS = list(BROKER_META.keys())


def infer_asset_class(ticker: str) -> str:
    """Infer a ticker/symbol's asset class from its format, so the auto-trader
    can automatically skip target accounts that don't support it (see
    auto_trader.py's _trade_target) instead of attempting — and erroring on —
    every target against every configured account regardless of fit. This is
    what lets an IG (forex-only) account sit in control.account_ids alongside
    equity accounts permanently without generating wasted API calls or error
    noise on equity-only cycles ("crosstalk"), rather than requiring it to be
    manually added/removed before each restart.

    Polygon convention (see universe.py's crypto/forex CSVs, already used
    directly as scan/backtest tickers): crypto is "X:BTCUSD"-style, forex is
    "C:EURUSD"-style. A resolved IG epic (e.g. "CS.D.EURUSD.MINI.IP" — what an
    actual forex auto-trader target uses, once one exists, not a backtest
    ticker) is also forex. Anything else defaults to equity, the overwhelming
    common case (plain tickers like "AAPL").
    """
    if ticker.startswith("X:"):
        return "crypto"
    if ticker.startswith("C:") or ticker.startswith("CS.D.") or ".CFD." in ticker or ".MINI." in ticker:
        return "forex"
    return "equity"


def account_asset_class(account: dict) -> str:
    """The single asset class THIS linked account actually trades — the
    authoritative answer for UI sectioning/compatibility checks, as opposed
    to BROKER_META[broker]["asset_classes"] which is what the broker TYPE is
    capable of. Every broker except IBKR has exactly one asset class per
    account (BROKER_META already models that as a 1-element set); IBKR is the
    sole broker where two linked accounts of the same broker type can differ
    (a locally-run gateway can be pointed at either an equity or forex sub-
    account), so it alone reads a per-account conn_params override. A missing
    override defaults to "equity" — matching build_broker_accounts()'s own
    default, so every account linked before this field existed (including the
    real "IBKR Paper" account) keeps behaving exactly as before.

    Takes the same dict shape list_accounts() returns.
    """
    broker = account["broker"]
    if BROKER_META.get(broker, {}).get("uses_gateway"):
        return account.get("conn_params", {}).get("asset_class", "equity")
    classes = BROKER_META.get(broker, {}).get("asset_classes", frozenset({"equity"}))
    return next(iter(classes))


def _load_raw() -> list[dict]:
    if not ACCOUNTS_PATH.exists():
        return []
    return json.loads(ACCOUNTS_PATH.read_text(encoding="utf-8"))


def _save_raw(accounts: list[dict]) -> None:
    atomic_write_text(ACCOUNTS_PATH, json.dumps(accounts, indent=2))


def _keyring_key(account_id: str, field: str) -> str:
    return f"{account_id}:{field}"


def list_accounts() -> list[dict]:
    """Return account metadata — safe to render in the UI, no secrets included.
    Gateway brokers (IBKR) carry non-secret `conn_params` instead of a masked key."""
    result = []
    for a in _load_raw():
        item = {
            "id": a["id"],
            "nickname": a["nickname"],
            "broker": a["broker"],
            "is_paper": a["is_paper"],
            "api_key_masked": f"****{a['api_key_last4']}" if a.get("api_key_last4") else "(not set)",
        }
        if a.get("conn_params"):
            item["conn_params"] = a["conn_params"]  # host/port/client_id/ibkr_account — non-secret
        result.append(item)
    return result


def add_account(
    nickname: str,
    broker: str,
    is_paper: bool,
    api_key: str = "",
    secret_key: str = "",
    conn_params: dict | None = None,
    extra_cred: str = "",
) -> str:
    if broker not in SUPPORTED_BROKERS:
        raise ValueError(f"Unsupported broker '{broker}'. Supported: {SUPPORTED_BROKERS}")

    if not BROKER_META[broker]["supports_paper"]:
        is_paper = False  # no verified sandbox for this broker — never claim paper safety we can't back up

    accounts = _load_raw()
    account_id = str(uuid.uuid4())
    row: dict = {"id": account_id, "nickname": nickname, "broker": broker, "is_paper": is_paper}

    if BROKER_META[broker].get("extra_cred_field"):
        keyring.set_password(KEYRING_SERVICE, _keyring_key(account_id, "extra"), extra_cred)

    if BROKER_META[broker].get("uses_gateway"):
        # IBKR: no secrets. Store only non-secret connection config as metadata —
        # nothing goes into the OS keyring for these accounts.
        cp = conn_params or {}
        asset_class = cp.get("asset_class", "equity")
        if asset_class not in IBKR_ASSET_CLASSES:
            raise ValueError(f"Unknown asset_class {asset_class!r}. Available: {IBKR_ASSET_CLASSES}")
        row["conn_params"] = {
            "host": cp.get("host", "127.0.0.1"),
            "port": int(cp.get("port", 4002)),
            "client_id": int(cp.get("client_id", 1)),
            "ibkr_account": cp.get("ibkr_account", ""),
            "asset_class": asset_class,
        }
        row["api_key_last4"] = None
    else:
        keyring.set_password(KEYRING_SERVICE, _keyring_key(account_id, "api_key"), api_key)
        keyring.set_password(KEYRING_SERVICE, _keyring_key(account_id, "secret_key"), secret_key)
        row["api_key_last4"] = api_key[-4:] if len(api_key) >= 4 else "*" * len(api_key)

    accounts.append(row)
    _save_raw(accounts)
    return account_id


def remove_account(account_id: str) -> None:
    for field in ("api_key", "secret_key", "extra"):
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
        broker = a["broker"]

        # Gateway brokers (IBKR) have no keyring secrets — build them from the
        # stored connection params and skip the secret lookup entirely.
        if BROKER_META.get(broker, {}).get("uses_gateway"):
            cp = a.get("conn_params") or {}
            obj = IBKRBroker(
                nickname=a["nickname"],
                host=cp.get("host", "127.0.0.1"),
                port=cp.get("port", 4002),
                client_id=cp.get("client_id", 1),
                ibkr_account=cp.get("ibkr_account", ""),
                is_paper=a["is_paper"],
                # Set via the Accounts tab's link form (an Equity/Forex radio on
                # the IBKR branch). Defaults to "equity" for any account linked
                # before that field existed, so nothing changes for them.
                asset_class=cp.get("asset_class", "equity"),
            )
            obj.account_id = a["id"]
            built.append(obj)
            continue

        api_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(a["id"], "api_key"))
        secret_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(a["id"], "secret_key"))
        if not api_key or not secret_key:
            raise ValueError(
                f"Credentials for account '{a['nickname']}' are missing from the OS keyring "
                "(deleted outside this app, or a different OS user account). Remove and re-link it."
            )

        if broker == "ig":
            username = keyring.get_password(KEYRING_SERVICE, _keyring_key(a["id"], "extra"))
            if not username:
                raise ValueError(
                    f"IG username for account '{a['nickname']}' is missing from the OS keyring "
                    "(deleted outside this app, or a different OS user account). Remove and re-link it."
                )
            obj = IGBroker(
                nickname=a["nickname"], username=username, password=secret_key,
                api_key=api_key, is_paper=a["is_paper"],
            )
        elif broker == "alpaca":
            obj = AlpacaBroker(nickname=a["nickname"], api_key=api_key, secret_key=secret_key, is_paper=a["is_paper"])
        elif broker == "oanda":
            # secret_key holds the account ID here, not a real secret — see
            # BROKER_META's "oanda" entry for why this reuses the standard
            # 2-field slot instead of a dedicated field.
            obj = OandaBroker(nickname=a["nickname"], api_key=api_key, account_id=secret_key, is_paper=a["is_paper"])
        elif broker == "coinbase":
            obj = CoinbaseBroker(
                nickname=a["nickname"], api_key=api_key, api_secret=secret_key,
                quote_currency=a.get("quote_currency", "USD"),
            )
        elif broker == "kraken":
            obj = KrakenBroker(nickname=a["nickname"], api_key=api_key, api_secret=secret_key)
        elif broker == "tastytrade":
            obj = TastytradeBroker(
                nickname=a["nickname"],
                refresh_token=api_key,
                provider_secret=secret_key,
                is_paper=a["is_paper"],
            )
        elif broker == "etoro":
            obj = EToroBroker(nickname=a["nickname"], api_key=api_key, user_key=secret_key, is_paper=a["is_paper"])
        else:
            raise ValueError(f"Unsupported broker '{broker}' for account {a['nickname']}")

        obj.account_id = a["id"]
        built.append(obj)
    return built
