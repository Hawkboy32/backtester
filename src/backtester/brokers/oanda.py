"""OANDA v20 REST API broker — real order execution, not just the read-only
candle data OandaDataClient (backtester.oanda_data) already provides.

Scope, agreed 2026-08-08 (see CLAUDE_NOTES.txt "PENDING IDEAS" entry): PAPER
(practice environment) only for now — going live is a separate, later
decision, matching this project's cautious-by-default posture
([[trading-bot-goals-and-risk-posture]]). `is_paper` still selects between
OANDA's practice/live base URLs (same pattern as every other broker here), so
going live later is a config change, not a code change — but nothing in this
project should set is_paper=False for an OANDA account without a deliberate,
separate decision.

Credentials: an OANDA API token (Bearer auth, same as OandaDataClient) PLUS an
account ID — unlike OandaDataClient's candle endpoints, every order/position/
account endpoint here is scoped to one specific account. Reuses
accounts.py's normal 2-field keyring flow (api_key=token, secret_key=account
ID) rather than needing a third credential field.

Order model: OANDA nets orders per-instrument automatically, so opening and
closing both go through the SAME POST /orders call — a BUY sends positive
units, a SELL sends negative units. While flat, a BUY opens a long and a SELL
opens a short (both real order paths here — see account_position_modes in
auto_trader_state.py for which accounts are actually allowed to). While
holding a position, the OPPOSITE side closes it — a SELL closes an existing
long, a BUY closes (covers) an existing short — always sized to the exact
held quantity, never freshly computed (see auto_trader.py's close-side
handling). No separate close-position endpoint needed, unlike ig.py's IG
integration.
"""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position
from backtester.brokers.ibkr import _forex_clock_heuristic
from backtester.oanda_data import LIVE_BASE_URL, PRACTICE_BASE_URL, OandaError, _to_oanda_instrument


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


class OandaBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, account_id: str, is_paper: bool = True):
        self.nickname = nickname
        self.is_paper = is_paper
        self._api_key = api_key
        self._account_id = account_id
        self.base_url = PRACTICE_BASE_URL if is_paper else LIVE_BASE_URL
        self.session = requests.Session()

    @property
    def supports_fractional_shares(self) -> bool:
        """OANDA's v20 API rejects orders with too much precision for the
        instrument — confirmed live 2026-08-10: "The units specified contain
        more precision than is allowed for the Order's instrument" — standard
        forex pairs trade in whole units, not fractional ones. Without this
        override, compute_qty_for_account's floor-to-whole-shares step never
        ran (base.py's default is True), so a %-equity/fixed-dollar sizing
        target sailed through with 4-decimal-place precision straight into a
        rejected order. Same pattern as ibkr.py's per-broker override, just
        the opposite conclusion — IBKR's forex (CASH) leg allows fractional
        units, OANDA's standard order path here doesn't."""
        return False

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{self.base_url}{path}"
        resp = self.session.request(method, url, headers=self._headers(), timeout=30, **kwargs)
        try:
            payload = resp.json()
        except ValueError:
            payload = {}
        if not resp.ok:
            message = payload.get("errorMessage") or payload.get("rejectReason") or f"HTTP {resp.status_code}"
            raise OandaError(f"OANDA API error: {message}")
        return payload

    def get_account_snapshot(self) -> AccountSnapshot:
        payload = self._request("GET", f"/v3/accounts/{self._account_id}/summary")
        account = payload.get("account", {})
        return AccountSnapshot(
            account_id=self._account_id,
            equity=_num(account.get("NAV")),
            cash=_num(account.get("balance")),
            buying_power=_num(account.get("marginAvailable")),
            is_paper=self.is_paper,
        )

    def get_positions(self) -> list[Position]:
        payload = self._request("GET", f"/v3/accounts/{self._account_id}/openPositions")
        positions: list[Position] = []
        for row in payload.get("positions", []):
            instrument = str(row.get("instrument", ""))
            ticker = "C:" + instrument.replace("_", "")
            long_units = _num(row.get("long", {}).get("units"))
            short_units = _num(row.get("short", {}).get("units"))
            # Most accounts here are still long-only by config
            # (account_position_modes), but short_units is a real, expected
            # value now for one explicitly opted into short_only/long_short
            # — reported either way, not assumed zero.
            side_data = row.get("long", {}) if long_units != 0 else row.get("short", {})
            qty = long_units if long_units != 0 else abs(short_units)
            if qty == 0:
                continue
            entry = _num(side_data.get("averagePrice"))
            unrealized = _num(side_data.get("unrealizedPL"))
            positions.append(
                Position(
                    ticker=ticker,
                    qty=qty,
                    side="long" if long_units != 0 else "short",
                    avg_entry_price=entry,
                    current_price=None,  # not in this payload; unrealized_pl below is authoritative
                    market_value=qty * entry,
                    unrealized_pl=unrealized,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        # Same gap as ig.py/coinbase.py/kraken.py: OANDA's transaction-history
        # endpoints exist but reconstructing a portfolio-equity time series
        # from them hasn't been verified against a real account yet. The
        # balances chart already skips brokers that raise this.
        raise NotImplementedError(
            "OANDA's v20 API doesn't expose a ready-made equity time series through this "
            "client yet — only current balance (see get_account_snapshot)."
        )

    def get_market_clock(self) -> dict | None:
        # Same heuristic ig.py uses for forex CFDs — real spot forex hours,
        # which is exactly what OANDA trades (no separate CFD wrapper here).
        return _forex_clock_heuristic()

    def get_open_order_tickers(self) -> set[str]:
        # submit_market_order below only ever places MARKET orders, which
        # OANDA fills synchronously in the same response (fillTransaction
        # present or the order is cancelled) — no separate pending state to
        # dedupe against, same reasoning as ig.py's own market-order-only flow.
        return set()

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        try:
            instrument = _to_oanda_instrument(ticker)
        except OandaError as e:
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))

        units = qty if side is OrderSide.BUY else -qty
        order: dict = {
            "type": "MARKET",
            "instrument": instrument,
            # Always whole units, never str(units)'s raw float precision —
            # supports_fractional_shares=False makes compute_qty_for_account
            # floor to a whole number for NEW orders, but a close-order's qty
            # comes from a previously-recorded position (get_positions()),
            # which predates this fix or could still carry float noise, so
            # round here too rather than trusting every caller.
            "units": f"{round(units):.0f}",
            "timeInForce": "FOK",  # fill-or-kill — matches "market order" semantics, no partial-fill surprise
            "positionFill": "DEFAULT",  # nets against any existing opposite position automatically
        }
        if side is OrderSide.BUY and take_profit_price is not None:
            order["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}
        if side is OrderSide.BUY and stop_loss_price is not None:
            order["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}

        try:
            payload = self._request(
                "POST", f"/v3/accounts/{self._account_id}/orders", json={"order": order}
            )
        except OandaError as e:
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))

        fill = payload.get("orderFillTransaction")
        if fill is None:
            cancel = payload.get("orderCancelTransaction", {})
            reason = cancel.get("reason", "order was not filled")
            return OrderResult(
                account_nickname=self.nickname, success=False,
                broker_order_id=payload.get("orderCreateTransaction", {}).get("id"),
                error=str(reason),
            )
        return OrderResult(
            account_nickname=self.nickname, success=True,
            broker_order_id=fill.get("id"),
            filled_qty=abs(_num(fill.get("units"), qty)),
            filled_avg_price=_num(fill.get("price")) or None,
        )
