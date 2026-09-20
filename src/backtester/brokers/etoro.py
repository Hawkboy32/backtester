"""eToro Public API broker — added 2026-09-16 as the read side of a planned
copy-trading signal source: poll a Popular Investor's live portfolio (via
their public users-info endpoints, not covered by this class) and turn
position changes into BUY/SELL signals through the normal roster/execution
pipeline, the same way any other strategy's signals do. This class is just
the account-linking half (snapshot/positions/orders on the USER'S OWN eToro
account) — the investor-watching/signal-diffing piece is separate, not built
yet.

Endpoints and request shapes below are taken directly from eToro's own API
reference (api-portal.etoro.com) as of 2026-09-16, not guessed — but unlike
every other broker in this file, none of it has been exercised against a
real eToro account yet, and the reference didn't show a response body for
create-an-order at all (submit_market_order's result-parsing below is a
best-effort guess at field names, not verified). Confirm every response
shape on first real call — get_account_snapshot/get_positions first (read-
only, safe on a Demo key), submit_market_order last and only once the read
side is confirmed working — and fix anything that doesn't match. Same
"verify against the real thing" discipline as everywhere else in this file,
just not done yet because there was no key to test with when this was
written.

Credentials: two keys from Settings -> Trading -> API Key Management on
eToro's own site, generated per-environment (Demo and Real are SEPARATE
keys, not one key with a mode flag) — reuses accounts.py's standard 2-field
keyring flow (api_key = Public API Key / x-api-key, secret_key = User Key /
x-user-key), same as every non-IG/non-IBKR broker here.

Known gap: get_positions() exposes eToro's raw numeric instrumentID as the
ticker (e.g. "ETORO:1002") rather than a real symbol — eToro's positions
payload doesn't carry a symbol, only an instrument ID, and resolving that to
a tradeable symbol needs a separate instrument-lookup call this class
doesn't make yet. Fine for now (nothing routes an eToro position through
infer_asset_class or a scan ticker yet), but a real blocker before this
account could be used for anything beyond read-only balance/position display.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import requests

from backtester.brokers.base import (
    AccountSnapshot,
    BrokerAccount,
    EquityPoint,
    OrderResult,
    OrderSide,
    Position,
)

BASE_URL = "https://public-api.etoro.com/api/v1"
ORDERS_URL = "https://public-api.etoro.com/api/v2/trading/execution/orders"


class EToroError(RuntimeError):
    """An eToro API call failed or returned an unexpected shape."""


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class EToroBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, user_key: str, is_paper: bool = True):
        self.nickname = nickname
        self.is_paper = is_paper
        self._api_key = api_key
        self._user_key = user_key
        # eToro's own vocabulary is "demo"/"real", not "paper"/"live" — kept
        # as an internal detail so the rest of this app never has to know.
        self._env = "demo" if is_paper else "real"
        self.session = requests.Session()

    def _headers(self) -> dict:
        return {
            "x-request-id": str(uuid.uuid4()),
            "x-api-key": self._api_key,
            "x-user-key": self._user_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, **kwargs) -> dict:
        resp = self.session.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if not resp.ok:
            message = payload.get("message") or payload.get("error") or f"HTTP {resp.status_code}"
            raise EToroError(f"eToro API error: {message}")
        return payload

    def get_account_snapshot(self) -> AccountSnapshot:
        payload = self._request("GET", f"{BASE_URL}/trading/info/{self._env}/pnl")
        portfolio = payload.get("clientPortfolio", {})
        credit = _num(portfolio.get("credit"))  # cash/available balance, per eToro's own docs
        unrealized = _num(portfolio.get("unrealizedPnL"))
        return AccountSnapshot(
            account_id=self._user_key[-8:],  # eToro's docs don't expose a separate numeric account id here
            equity=credit + unrealized,
            cash=credit,
            buying_power=credit,  # eToro's leverage-dependent buying power isn't in this payload — see module docstring
            is_paper=self.is_paper,
        )

    def get_positions(self) -> list[Position]:
        payload = self._request("GET", f"{BASE_URL}/trading/info/{self._env}/portfolio")
        positions: list[Position] = []
        for row in payload.get("clientPortfolio", {}).get("positions", []):
            units = _num(row.get("units"))
            if units == 0:
                continue
            # See module docstring: no real symbol in this payload, only a
            # numeric instrument id. Synthetic ticker until instrument-lookup
            # is wired up — deliberately namespaced so it can never collide
            # with (or be mistaken for) a real Polygon-style ticker elsewhere.
            ticker = f"ETORO:{row.get('instrumentID')}"
            entry = _num(row.get("openRate"))
            current_value = _num(row.get("unitsBaseValueDollars")) or units * entry
            positions.append(
                Position(
                    ticker=ticker,
                    qty=units,
                    side="long" if row.get("isBuy", True) else "short",
                    avg_entry_price=entry,
                    current_price=(current_value / units) if units else None,
                    market_value=current_value,
                    unrealized_pl=_num(row.get("unrealizedPnL", {}).get("pnL")) if isinstance(row.get("unrealizedPnL"), dict) else 0.0,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        # Same gap as ig.py/oanda.py/coinbase.py/kraken.py: no equity time
        # series verified against this API yet — see those modules' own
        # identical NotImplementedError for the established pattern this
        # follows. The dashboard's balances chart already skips brokers that
        # raise this rather than treating it as an error.
        raise NotImplementedError(
            "eToro's API doesn't have a verified equity time-series endpoint through this "
            "client yet — only current balance (see get_account_snapshot)."
        )

    def get_open_order_tickers(self) -> set[str]:
        return set()

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        if not ticker.startswith("ETORO:"):
            return OrderResult(
                account_nickname=self.nickname, success=False,
                error=f"eToro only trades its own instrument-id tickers (ETORO:<id>); got {ticker!r}.",
            )
        instrument_id = ticker.removeprefix("ETORO:")

        order: dict = {
            "action": "open",
            "transaction": "buy" if side is OrderSide.BUY else "sellShort",
            "instrumentId": int(instrument_id),
            "orderType": "mkt",
            "units": qty,
            "leverage": 1,
        }
        # eToro requires a stop loss on any sellShort order regardless of
        # leverage (per their own API reference) — without one, a short
        # entry is rejected outright rather than opened unprotected.
        if side is OrderSide.SELL and stop_loss_price is None:
            return OrderResult(
                account_nickname=self.nickname, success=False,
                error="eToro requires stop_loss_price on a short (sellShort) order.",
            )
        if stop_loss_price is not None:
            order["stopLossRate"] = stop_loss_price
        if take_profit_price is not None:
            order["takeProfitRate"] = take_profit_price

        try:
            payload = self._request("POST", ORDERS_URL, json=order)
        except EToroError as e:
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))

        return OrderResult(
            account_nickname=self.nickname, success=True,
            broker_order_id=str(payload.get("orderId") or payload.get("id") or ""),
            filled_qty=_num(payload.get("units"), qty),
            filled_avg_price=_num(payload.get("openRate")) or None,
        )
