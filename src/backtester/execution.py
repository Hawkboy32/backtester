"""Multi-account order execution and position sizing.

This module only ever runs when explicitly called — nothing in the scanner,
strategies, or "live feed" triggers this automatically. Submitting an order
is always a deliberate, separate action taken by whoever calls
execute_order_across_accounts (in this project, that's the dashboard's Trade
Execution tab, gated behind an explicit confirm step).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from backtester.brokers.base import BrokerAccount, OrderResult, OrderSide


class SizingMode(Enum):
    FIXED_SHARES = "fixed_shares"
    PCT_EQUITY = "pct_equity"
    FIXED_DOLLARS = "fixed_dollars"


@dataclass
class AccountOrder:
    account: BrokerAccount
    qty: float
    take_profit_price: float | None = None
    stop_loss_price: float | None = None


def compute_qty_for_account(
    account: BrokerAccount,
    reference_price: float,
    sizing_mode: SizingMode,
    sizing_value: float,
) -> float:
    """Translate a sizing rule into a share quantity for one account.

    reference_price is a price hint you supply (e.g. the last quote you saw)
    used only to convert a %-of-equity or fixed-dollar target into a share
    count — it is not used for execution, which remains a market order at
    whatever price actually fills.
    """
    if sizing_mode is SizingMode.FIXED_SHARES:
        return sizing_value

    if reference_price <= 0:
        raise ValueError("reference_price must be positive to size by equity % or dollar amount")

    if sizing_mode is SizingMode.PCT_EQUITY:
        snapshot = account.get_account_snapshot()
        dollars = snapshot.equity * (sizing_value / 100)
    elif sizing_mode is SizingMode.FIXED_DOLLARS:
        dollars = sizing_value
    else:
        raise ValueError(f"Unknown sizing mode: {sizing_mode}")

    return round(dollars / reference_price, 4)


def execute_order_across_accounts(
    orders: list[AccountOrder],
    ticker: str,
    side: OrderSide,
    max_workers: int = 8,
) -> list[OrderResult]:
    """Submit an order to every given account concurrently, one qty (and
    optional bracket prices) per account.

    Each account's result (success or failure) is independent — one
    account failing doesn't stop or roll back the others. Returns one
    OrderResult per account, in the same order as `orders`.
    """
    if not orders:
        return []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(orders))) as pool:
        futures = [
            pool.submit(
                o.account.submit_market_order,
                ticker,
                side,
                o.qty,
                o.take_profit_price,
                o.stop_loss_price,
            )
            for o in orders
        ]
        return [future.result() for future in futures]
