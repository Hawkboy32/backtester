"""Multi-account order execution and position sizing.

This module only ever runs when explicitly called — nothing in the scanner,
strategies, or "live feed" triggers this automatically. Submitting an order
is always a deliberate, separate action taken by whoever calls
execute_order_across_accounts (in this project, that's the dashboard's Trade
Execution tab, gated behind an explicit confirm step).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from backtester.brokers.base import BrokerAccount, OrderResult, OrderSide


class SizingMode(Enum):
    FIXED_SHARES = "fixed_shares"
    PCT_EQUITY = "pct_equity"
    FIXED_DOLLARS = "fixed_dollars"


def sliding_pct_equity(equity: float, start_pct: float, target_pct: float, floor_notional: float = 1.0) -> float:
    """The %-of-equity sizing rate for a small account that's sliding from an
    aggressive starting rate down toward the account's real target rate as it
    grows — replaces a hard equity threshold (auto_trader_state.py's earlier
    account_sizing_overrides shape) with a smooth interpolation, added
    2026-08-09 at the user's own suggestion after the earlier threshold was
    found to not actually line up with when the target rate clears a real
    broker's minimum order size.

    Both ends of the slide are DERIVED from floor_notional (the broker's real
    minimum notional per order, e.g. Alpaca's ~$1 for fractional shares), not
    picked arbitrarily:
      - e_lo = floor_notional / (start_pct/100): the equity at which
        start_pct ITSELF barely clears the floor. Below this even the
        aggressive starting rate can't place a valid order, so the rate
        stays pinned at start_pct rather than trying to go higher.
      - e_hi = floor_notional / (target_pct/100): the equity at which the
        account's real target_pct clears the floor on its own - past this
        point there's no more reason to boost sizing above target_pct, so
        the slide is done and this function is no longer even needed by the
        caller (target_pct applies directly).
    Interpolated log-linearly in equity between the two (so the visually
    "smooth" decline happens over the actual order-of-magnitude range that
    matters, e.g. $2 to $100, not skewed by a linear equity axis), with the
    rate itself interpolated linearly between start_pct and target_pct.
    """
    if target_pct <= 0 or start_pct <= target_pct:
        return target_pct
    e_lo = floor_notional / (start_pct / 100)
    e_hi = floor_notional / (target_pct / 100)
    if equity <= e_lo:
        return start_pct
    if equity >= e_hi:
        return target_pct
    frac = (math.log(equity) - math.log(e_lo)) / (math.log(e_hi) - math.log(e_lo))
    return start_pct + (target_pct - start_pct) * frac


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

    Floors to a whole share for any account whose broker doesn't accept
    fractional-sized orders (account.supports_fractional_shares == False,
    e.g. IBKR equities) — otherwise %-of-equity/fixed-dollar sizing routinely
    produces a fractional quantity that broker's API rejects outright.
    """
    if sizing_mode is SizingMode.FIXED_SHARES:
        qty = sizing_value
    else:
        if reference_price <= 0:
            raise ValueError("reference_price must be positive to size by equity % or dollar amount")

        if sizing_mode is SizingMode.PCT_EQUITY:
            snapshot = account.get_account_snapshot()
            dollars = snapshot.equity * (sizing_value / 100)
        elif sizing_mode is SizingMode.FIXED_DOLLARS:
            dollars = sizing_value
        else:
            raise ValueError(f"Unknown sizing mode: {sizing_mode}")

        qty = round(dollars / reference_price, 4)

    if not account.supports_fractional_shares and qty != math.floor(qty):
        qty = float(math.floor(qty))
        if qty <= 0:
            raise ValueError(
                f"{sizing_value} ({sizing_mode.value}) at reference price {reference_price} floors to 0 "
                f"whole shares on {account.nickname} — this broker doesn't accept fractional orders "
                "and the sizing target doesn't cover even 1 share. Raise the sizing value or pick a "
                "cheaper ticker."
            )
    return qty


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
