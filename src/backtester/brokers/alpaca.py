"""Alpaca broker implementation, backed by the official alpaca-py SDK.

Defaults to Alpaca's paper-trading endpoint. An account only submits real
orders if is_paper=False was explicitly set when it was linked.
"""

from __future__ import annotations

from datetime import datetime, timezone

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    MarketOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
)

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position


class AlpacaBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, secret_key: str, is_paper: bool = True):
        self.nickname = nickname
        self.is_paper = is_paper
        self._client = TradingClient(api_key=api_key, secret_key=secret_key, paper=is_paper)

    def get_account_snapshot(self) -> AccountSnapshot:
        account = self._client.get_account()
        return AccountSnapshot(
            account_id=str(account.id),
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            is_paper=self.is_paper,
        )

    def get_positions(self) -> list[Position]:
        positions = self._client.get_all_positions()
        return [
            Position(
                ticker=p.symbol,
                qty=float(p.qty),
                side=p.side.value if hasattr(p.side, "value") else str(p.side),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price) if p.current_price is not None else None,
                market_value=float(p.market_value) if p.market_value is not None else 0.0,
                unrealized_pl=float(p.unrealized_pl) if p.unrealized_pl is not None else 0.0,
            )
            for p in positions
        ]

    def get_market_clock(self) -> dict | None:
        clock = self._client.get_clock()
        return {"is_open": clock.is_open, "next_open": clock.next_open}

    def get_open_order_tickers(self) -> set[str]:
        orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {o.symbol for o in orders}

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        history = self._client.get_portfolio_history(
            GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
        )
        timestamps = history.timestamp or []
        equity = history.equity or []
        return [
            EquityPoint(timestamp=datetime.fromtimestamp(ts, tz=timezone.utc), equity=eq)
            for ts, eq in zip(timestamps, equity)
            if eq is not None
        ]

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        try:
            order_kwargs = dict(
                symbol=ticker,
                qty=qty,
                side=AlpacaOrderSide.BUY if side is OrderSide.BUY else AlpacaOrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            if take_profit_price is not None or stop_loss_price is not None:
                order_kwargs["order_class"] = OrderClass.BRACKET
                if take_profit_price is not None:
                    order_kwargs["take_profit"] = TakeProfitRequest(limit_price=take_profit_price)
                if stop_loss_price is not None:
                    order_kwargs["stop_loss"] = StopLossRequest(stop_price=stop_loss_price)

            request = MarketOrderRequest(**order_kwargs)
            order = self._client.submit_order(order_data=request)
            return OrderResult(
                account_nickname=self.nickname,
                success=True,
                broker_order_id=str(order.id),
                filled_qty=float(order.filled_qty) if order.filled_qty is not None else None,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
            )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
