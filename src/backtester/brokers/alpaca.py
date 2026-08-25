"""Alpaca broker implementation, backed by the official alpaca-py SDK.

Defaults to Alpaca's paper-trading endpoint. An account only submits real
orders if is_paper=False was explicitly set when it was linked.
"""

from __future__ import annotations

import time

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
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

from backtester.brokers.base import (
    AccountSnapshot, BrokerAccount, BrokerFee, EquityPoint, OrderResult, OrderSide, Position,
)

# How long to wait for a real fill price after submitting (see
# submit_market_order). ~3s total: long enough for a market order in a live
# session, short enough not to stall a 120s poll loop when the market is shut
# and the order is simply queued.
_FILL_POLL_ATTEMPTS = 6
_FILL_POLL_SECONDS = 0.5

_TIMESPAN_TO_UNIT = {
    "minute": TimeFrameUnit.Minute,
    "hour": TimeFrameUnit.Hour,
    "day": TimeFrameUnit.Day,
}


class AlpacaBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, secret_key: str, is_paper: bool = True):
        self.nickname = nickname
        self.is_paper = is_paper
        self._client = TradingClient(api_key=api_key, secret_key=secret_key, paper=is_paper)
        # Separate client, same key pair — Alpaca's trading and market-data
        # APIs are independent products under one account. Free/paper accounts
        # only have rights to the IEX feed (not the full SIP tape); the default
        # feed 403s on "recent" data ("subscription does not permit querying
        # recent SIP data") unless IEX is requested explicitly everywhere below.
        self._data_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def get_account_snapshot(self) -> AccountSnapshot:
        account = self._client.get_account()
        return AccountSnapshot(
            account_id=str(account.id),
            equity=float(account.equity),
            cash=float(account.cash),
            buying_power=float(account.buying_power),
            is_paper=self.is_paper,
        )

    def get_fees(self, limit: int = 100) -> list[BrokerFee]:
        """Every FEE activity Alpaca has charged, newest first.

        Hits /v2/account/activities directly with requests rather than through
        alpaca-py: the installed SDK has no GetAccountActivitiesRequest, and
        this is a plain authenticated GET.

        Two very different things both arrive as FEE and both matter:
          - REG / TAF / CAT — per-sell regulatory fees, ~$0.01 each
          - "Funding Wallet incoming alpaca conversion fee" — the GBP->USD
            charge on a DEPOSIT, which is far larger (~1.5%) and has nothing
            to do with trading at all
        Kept as separate line items so the app can show which is which.
        """
        base = "https://paper-api.alpaca.markets" if self.is_paper else "https://api.alpaca.markets"
        headers = {
            "APCA-API-KEY-ID": self._client._api_key,
            "APCA-API-SECRET-KEY": self._client._secret_key,
        }
        resp = requests.get(
            f"{base}/v2/account/activities",
            headers=headers, params={"page_size": min(limit, 100)}, timeout=30,
        )
        resp.raise_for_status()
        fees = []
        for a in resp.json():
            if a.get("activity_type") != "FEE":
                continue
            try:
                amount = float(a.get("net_amount", 0) or 0)
            except (TypeError, ValueError):
                continue
            desc = a.get("description", "") or ""
            # Alpaca doesn't label the conversion charge with a code, so
            # classify off its own description rather than inventing one.
            kind = "CONVERSION" if "conversion" in desc.lower() else desc.split()[0] if desc else "FEE"
            fees.append(BrokerFee(
                date=str(a.get("date") or a.get("transaction_time") or "")[:10],
                kind=kind, amount=amount, description=desc,
            ))
        return fees

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
        # next_close is what an end-of-session flatten needs; without it the
        # caller can only tell whether the market is open, not how long it has
        # left. Added alongside the protective exits (2026-08-14).
        return {
            "is_open": clock.is_open,
            "next_open": clock.next_open,
            "next_close": getattr(clock, "next_close", None),
        }

    def get_open_order_tickers(self) -> set[str]:
        orders = self._client.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        return {o.symbol for o in orders}

    def get_live_bars(
        self, ticker: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = "minute",
    ) -> pd.DataFrame:
        """Same-day-capable alternative to PolygonClient.get_aggregates(),
        shaped identically (columns: open/high/low/close/volume/vwap/
        transactions, indexed by UTC timestamp) so strategy code needs no
        changes at all. Exists because this account's Polygon plan has NO
        same-day intraday data at all — bars only appear the day after a
        session closes (confirmed 2026-08-05, see CLAUDE_NOTES.txt) — while
        Alpaca's own free IEX feed, on the SAME account already used for
        equities trading, returns live minute bars with no such gap and no
        rate-limit surprise (unlike IG's history endpoint — see notes).

        feed=DataFeed.IEX is required on every call, not just implied by
        account tier: the default feed 403s requesting "recent" data on a
        free/paper account ("subscription does not permit querying recent
        SIP data").
        """
        unit = _TIMESPAN_TO_UNIT.get(timespan, TimeFrameUnit.Minute)
        request = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(multiplier, unit),
            start=datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc),
            end=datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc) + timedelta(days=1),
            feed=DataFeed.IEX,
        )
        raw = self._data_client.get_stock_bars(request).df
        columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
        if raw.empty:
            return pd.DataFrame(columns=columns)
        # Single-symbol request still comes back with a (symbol, timestamp)
        # MultiIndex — drop the redundant symbol level.
        if isinstance(raw.index, pd.MultiIndex):
            raw = raw.droplevel("symbol")
        raw = raw.rename(columns={"trade_count": "transactions"})
        return raw[columns].sort_index()

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

            # submit_order returns IMMEDIATELY, before the fill — filled_qty=0
            # and filled_avg_price=None on essentially every Alpaca order.
            # Callers that recorded the trade from that response fell back to
            # the current BAR CLOSE as the fill price, which is not the price
            # actually paid. Measured on AlpacaLive 2026-08-14: that inflated
            # recorded realised P&L to +$0.22 against a true +$0.06, a 3.5x
            # overstatement (one Q trade alone was booked at 142.59 when it
            # actually filled at 141.81). Poll briefly for the real fill so
            # the recorded price is the price that happened.
            #
            # Bounded and best-effort: a market order normally fills in well
            # under a second during a session, and outside one it legitimately
            # stays queued — that returns unfilled, which the caller already
            # treats as "queued for the open" rather than an error.
            for _ in range(_FILL_POLL_ATTEMPTS):
                if order.filled_avg_price is not None and float(order.filled_qty or 0) > 0:
                    break
                time.sleep(_FILL_POLL_SECONDS)
                try:
                    order = self._client.get_order_by_id(order.id)
                except Exception:  # noqa: BLE001 — keep whatever we already have
                    break

            return OrderResult(
                account_nickname=self.nickname,
                success=True,
                broker_order_id=str(order.id),
                filled_qty=float(order.filled_qty) if order.filled_qty is not None else None,
                filled_avg_price=float(order.filled_avg_price) if order.filled_avg_price is not None else None,
            )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
