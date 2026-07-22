"""Coinbase Advanced Trade broker implementation, backed by the official
coinbase-advanced-py SDK.

No verified sandbox/paper mode — every linked Coinbase account always
connects to the real live API (accounts.py forces is_paper=False for this
broker). Spot trading only; `ticker` must be a Coinbase product_id like
"BTC-USD", not a bare asset code.

Bracket orders (take_profit_price/stop_loss_price) aren't implemented —
Coinbase's trigger-bracket order shape doesn't map cleanly onto a simple
"market buy now, attach TP/SL" without further verification against a real
account, so passing those raises a clear error instead of guessing.
"""

from __future__ import annotations

import uuid

from coinbase.rest import RESTClient

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position


def _get(obj, key):
    """Read `key` from a coinbase-advanced-py response field whether it's a
    plain dict, an attribute object, or an item-accessible object. The SDK is
    inconsistent — e.g. breakdown.portfolio_balances is a dict, but
    breakdown.spot_positions items are objects — so both styles must be handled."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        return None


def _money(field) -> float:
    """Extract a float from a Coinbase money field like {'value': '0', 'currency': 'GBP'}
    (dict or object), or a bare number. Returns 0.0 if absent/unparseable."""
    if field is None:
        return 0.0
    val = _get(field, "value") if not isinstance(field, (int, float)) else field
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


class CoinbaseBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, api_secret: str):
        self.nickname = nickname
        self.is_paper = False  # no verified sandbox for this broker
        self._client = RESTClient(api_key=api_key, api_secret=api_secret)

    def _default_portfolio_uuid(self) -> str:
        portfolios = self._client.get_portfolios().portfolios or []
        default = next((p for p in portfolios if p.type == "DEFAULT"), None)
        chosen = default or (portfolios[0] if portfolios else None)
        if not chosen:
            raise ValueError("No Coinbase portfolio found for this account")
        return chosen.uuid

    def get_account_snapshot(self) -> AccountSnapshot:
        portfolio_uuid = self._default_portfolio_uuid()
        breakdown = self._client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid).breakdown
        balances = _get(breakdown, "portfolio_balances")
        equity = _money(_get(balances, "total_balance"))
        cash = _money(_get(balances, "total_cash_equivalent_balance"))
        return AccountSnapshot(account_id=portfolio_uuid, equity=equity, cash=cash, buying_power=cash, is_paper=False)

    def get_positions(self) -> list[Position]:
        portfolio_uuid = self._default_portfolio_uuid()
        breakdown = self._client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid).breakdown
        positions = []
        for p in _get(breakdown, "spot_positions") or []:
            if _get(p, "is_cash"):
                continue  # skip fiat cash lines (e.g. GBP/USD) — not tradeable positions
            qty = float(_get(p, "total_balance_crypto") or 0.0)
            if qty == 0:
                continue  # dust-free: don't report zero-balance assets as positions
            market_value = float(_get(p, "total_balance_fiat") or 0.0)
            cost_basis = _money(_get(p, "cost_basis"))
            positions.append(
                Position(
                    ticker=_get(p, "asset"),
                    qty=qty,
                    side="long",
                    avg_entry_price=(cost_basis / qty) if cost_basis and qty else 0.0,
                    current_price=(market_value / qty) if qty else None,
                    market_value=market_value,
                    unrealized_pl=(market_value - cost_basis) if cost_basis else 0.0,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        raise NotImplementedError(
            "Coinbase's Advanced Trade API doesn't expose a historical portfolio-equity time "
            "series through this SDK — only current balances (see get_account_snapshot)."
        )

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        if take_profit_price is not None or stop_loss_price is not None:
            return OrderResult(
                account_nickname=self.nickname,
                success=False,
                error="Bracket orders (take-profit/stop-loss) aren't implemented for Coinbase yet.",
            )
        try:
            client_order_id = str(uuid.uuid4())
            if side is OrderSide.BUY:
                response = self._client.market_order_buy(
                    client_order_id=client_order_id, product_id=ticker, base_size=str(qty)
                )
            else:
                response = self._client.market_order_sell(
                    client_order_id=client_order_id, product_id=ticker, base_size=str(qty)
                )
            if not response.success:
                error_msg = response.error_response.error if response.error_response else "unknown error"
                return OrderResult(account_nickname=self.nickname, success=False, error=str(error_msg))
            return OrderResult(account_nickname=self.nickname, success=True, broker_order_id=response.order_id)
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
