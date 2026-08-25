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


@dataclass
class BrokerFee:
    """One fee the broker charged. `amount` is NEGATIVE (money leaving).

    `kind` is the broker's own code where it has one (Alpaca: REG/TAF/CAT
    regulatory fees, or a currency-conversion charge on funding); `description`
    is its human-readable line, kept verbatim rather than reworded so the app
    shows exactly what the broker says it charged for.
    """
    date: str
    kind: str
    amount: float
    description: str

    @property
    def is_funding(self) -> bool:
        """True for costs charged on money ENTERING the account (GBP->USD
        conversion), as opposed to costs of trading.

        The split matters because the two behave completely differently, and
        lumping them into one "fees" total hides that. Measured on AlpacaLive
        over 2026-08-11..13: trading fees were FIXED at $0.03/day while sell
        proceeds tripled ($5.69 -> $17.46), because every regulatory fee rounds
        up to a $0.01 minimum (below ~$360 per sell the true SEC fee is smaller
        than the floor). Conversion, by contrast, is ~1.5% of every deposit and
        scales forever. So trading cost is a fixed toll you outgrow; funding
        cost is a percentage you don't.
        """
        return self.kind.upper() == "CONVERSION" or "conversion" in self.description.lower()


def summarize_fees(fees: list[BrokerFee]) -> dict[str, float]:
    """Split a fee list into the two kinds that behave differently.

    Returns NEGATIVE amounts throughout (money leaving), matching BrokerFee,
    so callers can add these straight onto a gross P&L without sign juggling.
    Shared by the dashboard and the mobile backend so both classify identically
    rather than each re-deriving the rule.
    """
    trading = sum(f.amount for f in fees if not f.is_funding)
    funding = sum(f.amount for f in fees if f.is_funding)
    return {"trading": trading, "funding": funding, "total": trading + funding}


class BrokerAccount(ABC):
    """One linked, authenticated brokerage account."""

    nickname: str
    is_paper: bool
    account_id: str = ""  # the linked-account UUID from accounts.py; set by build_broker_accounts()
    # Whether this broker's API accepts a non-integer share quantity. True for
    # every broker in this project except IBKR (its API rejects fractional-
    # sized equity orders outright — "use the desktop version" — confirmed
    # live 2026-08-03). execution.compute_qty_for_account floors to a whole
    # share for any account where this is False, so %-of-equity/fixed-dollar
    # sizing never produces a quantity the broker will reject.
    supports_fractional_shares: bool = True

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

    def get_fees(self, limit: int = 100) -> list[BrokerFee]:
        """Fees the broker has actually charged, newest first.

        These never appear in live_trades.db — that only holds round-trip
        entry/exit prices, so its realised P&L is GROSS of costs. On a small
        account the difference is not academic: reconciling AlpacaLive on
        2026-08-14 showed $0.22 of gross trading profit against $0.90 of fees,
        of which $0.81 was GBP->USD conversion on deposits. Without surfacing
        these, the app reports a profit on an account that is actually down.

        NotImplementedError (not an empty list) where a broker has no uniform
        fee endpoint — the caller must be able to tell "this broker can't tell
        us" apart from "no fees charged".
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
