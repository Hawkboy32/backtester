"""Broker abstraction. Each concrete broker (Alpaca, and later others) implements
this interface so the execution engine and dashboard don't need to know which
broker they're talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class AccountSnapshot:
    account_id: str
    equity: float
    cash: float
    buying_power: float
    is_paper: bool


@dataclass
class Position:
    ticker: str
    qty: float
    side: str
    avg_entry_price: float
    current_price: float | None
    market_value: float
    unrealized_pl: float


@dataclass
class OrderResult:
    account_nickname: str
    success: bool
    broker_order_id: str | None = None
    filled_qty: float | None = None
    filled_avg_price: float | None = None
    error: str | None = None


@dataclass
class EquityPoint:
    timestamp: datetime
    equity: float


class BrokerAccount(ABC):
    """One linked, authenticated brokerage account."""

    nickname: str
    is_paper: bool
    account_id: str = ""  # the linked-account UUID from accounts.py; set by build_broker_accounts()

    @abstractmethod
    def get_account_snapshot(self) -> AccountSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> list[Position]:
        raise NotImplementedError

    @abstractmethod
    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        """Real historical account equity, not a projection. `period` and
        `timeframe` follow Alpaca's conventions (e.g. period='1M'/'3M'/'1Y',
        timeframe='1D'/'1H').
        """
        raise NotImplementedError

    def get_market_clock(self) -> dict | None:
        """{'is_open': bool, 'next_open': datetime} where the broker exposes a
        market clock; None where unsupported (e.g. 24/7 crypto venues). Not
        abstract — brokers without one just inherit the None default."""
        return None

    def get_open_order_tickers(self) -> set[str]:
        """Tickers with a currently-open (submitted-but-unfilled) order on this
        account. Used to avoid stacking duplicate orders on the same ticker
        before the first one fills — a filled position shows in get_positions,
        but a still-pending order does not. Default is an empty set (no dedup)
        for brokers that don't implement it; override per broker."""
        return set()

    @abstractmethod
    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        """Submit a market order. If take_profit_price and/or stop_loss_price
        are given, submits a bracket order (only meaningful for BUY orders on
        most brokers, including Alpaca).
        """
        raise NotImplementedError
