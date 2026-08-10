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


def _forex_usd_divisor(ticker: str, price: float) -> float:
    """What to divide a USD notional target by to get OANDA/IG-style
    base-currency units, for a Polygon-style forex ticker ("C:XXXYYY").

    When USD is the QUOTE currency (XXXUSD — GBPUSD, EURUSD, AUDUSD), 1 unit
    of the base currency is worth `price` USD, so units = dollars / price —
    the same share-count math as a stock. When USD is the BASE currency
    instead (USDXXX — USDJPY, USDCAD), 1 unit IS 1 USD of exposure directly,
    independent of price — dividing by price silently undersizes by a factor
    of roughly the price itself (confirmed live 2026-08-10: a ~$2000 target
    on USDJPY was sized to ~13 units / ~$13 notional instead, ~150x off — see
    CLAUDE_NOTES.txt). A pair involving neither currency (a cross pair, e.g.
    EURGBP) can't be converted to USD notional from its own price alone —
    raises rather than silently guessing.
    """
    pair = ticker[2:] if ticker.startswith("C:") else ticker
    base, quote = pair[:3], pair[3:]
    if quote == "USD":
        return price
    if base == "USD":
        return 1.0
    raise ValueError(
        f"{ticker}: neither currency is USD — dollar-based sizing (%-equity or fixed-dollar) "
        "can't convert this cross pair to USD notional from its own price alone. Use "
        "SizingMode.FIXED_SHARES (literal units) for this ticker instead."
    )


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
    ticker: str,
) -> float:
    """Translate a sizing rule into a share/unit quantity for one account.

    reference_price is a price hint you supply (e.g. the last quote you saw)
    used only to convert a %-of-equity or fixed-dollar target into a share
    count — it is not used for execution, which remains a market order at
    whatever price actually fills.

    ticker is required (not just for logging) — a forex ticker ("C:XXXYYY")
    needs currency-pair-aware division to convert a USD notional target into
    OANDA/IG-style base-currency units correctly (see _forex_usd_divisor's
    own docstring for the real bug this fixes: dividing by price
    unconditionally silently undersized USD-base pairs like USDJPY by
    ~150x). Equities/crypto tickers are unaffected — same dollars/price
    share-count math as before.

    Floors to a whole share/unit for any account whose broker doesn't accept
    fractional-sized orders (account.supports_fractional_shares == False,
    e.g. IBKR equities, OANDA forex) — otherwise %-of-equity/fixed-dollar
    sizing routinely produces a fractional quantity that broker's API
    rejects outright.
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

        divisor = _forex_usd_divisor(ticker, reference_price) if ticker.startswith("C:") else reference_price

        qty = round(dollars / divisor, 4)

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
