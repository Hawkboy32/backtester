"""Tastytrade broker implementation, backed by the official (async-only)
tastytrade Python SDK. Wraps each async SDK call with asyncio.run() since
the rest of this app is synchronous.

Auth is OAuth2 with two long-lived secrets you generate once in Tastytrade's
developer portal: a "provider secret" (client secret) and a personal
"refresh token" — not a simple API key/secret pair like the other brokers.
Has a real sandbox (`is_test=True`, Tastytrade's cert environment), so
paper/live both work like Alpaca.

Bracket orders (take_profit_price/stop_loss_price) aren't implemented —
Tastytrade's OTOCOOrder shape looks purpose-built for this, but the exact
sub-order construction wasn't verified against a real account, so this
raises a clear error instead of guessing.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from tastytrade import Account, Session
from tastytrade.order import InstrumentType, Leg, MarketOrder, OrderAction, OrderTimeInForce

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position


class TastytradeBroker(BrokerAccount):
    def __init__(self, nickname: str, refresh_token: str, provider_secret: str, is_paper: bool = True):
        self.nickname = nickname
        self.is_paper = is_paper
        self._session = Session(provider_secret=provider_secret, refresh_token=refresh_token, is_test=is_paper)

    async def _get_account(self) -> Account:
        result = await Account.get(self._session)
        return result[0] if isinstance(result, list) else result

    def get_account_snapshot(self) -> AccountSnapshot:
        async def _run():
            account = await self._get_account()
            balances = await account.get_balances(self._session)
            return account, balances

        account, balances = asyncio.run(_run())
        equity = float(balances.margin_equity) if balances.margin_equity is not None else 0.0
        cash = float(balances.cash_balance) if balances.cash_balance is not None else 0.0
        buying_power = float(balances.equity_buying_power) if balances.equity_buying_power is not None else cash
        return AccountSnapshot(
            account_id=account.account_number, equity=equity, cash=cash, buying_power=buying_power,
            is_paper=self.is_paper,
        )

    def get_positions(self) -> list[Position]:
        async def _run():
            account = await self._get_account()
            return await account.get_positions(self._session)

        raw_positions = asyncio.run(_run())
        positions = []
        for p in raw_positions:
            qty = float(p.quantity or 0)
            avg_entry = float(p.average_open_price) if p.average_open_price is not None else 0.0
            current = float(p.mark_price) if p.mark_price is not None else (float(p.mark) if p.mark is not None else None)
            multiplier = float(p.multiplier or 1)
            market_value = qty * current * multiplier if current is not None else 0.0
            unrealized_pl = (current - avg_entry) * qty * multiplier if current is not None else 0.0
            positions.append(
                Position(
                    ticker=p.symbol,
                    qty=qty,
                    side=str(p.quantity_direction),
                    avg_entry_price=avg_entry,
                    current_price=current,
                    market_value=market_value,
                    unrealized_pl=unrealized_pl,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        time_back_map = {"1W": "1w", "1M": "1m", "3M": "3m", "1Y": "1y"}
        time_back = time_back_map.get(period.upper(), "1m")

        async def _run():
            account = await self._get_account()
            return await account.get_net_liquidating_value_history(self._session, time_back=time_back)

        history = asyncio.run(_run())
        points = []
        for h in history:
            try:
                ts = datetime.fromisoformat(h.time.replace("Z", "+00:00"))
            except ValueError:
                continue
            points.append(EquityPoint(timestamp=ts, equity=float(h.close)))
        return points

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
                error="Bracket orders (take-profit/stop-loss) aren't implemented for Tastytrade yet.",
            )

        async def _run():
            account = await self._get_account()
            leg = Leg(
                instrument_type=InstrumentType.EQUITY,
                symbol=ticker,
                action=OrderAction.BUY if side is OrderSide.BUY else OrderAction.SELL,
                quantity=Decimal(str(qty)),
            )
            order = MarketOrder(time_in_force=OrderTimeInForce.DAY, legs=[leg])
            return await account.place_order(self._session, order, dry_run=False)

        try:
            response = asyncio.run(_run())
            if response.errors:
                return OrderResult(account_nickname=self.nickname, success=False, error="; ".join(map(str, response.errors)))
            return OrderResult(account_nickname=self.nickname, success=True, broker_order_id=str(response.order.id))
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
