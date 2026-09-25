"""Regression test for the AlpacaLive insufficient-buying-power pattern
(execution.py's BUYING_POWER_BUFFER) - recurred repeatedly 2026-08-31 through
2026-09-21, always rejected by a few cents because compute_qty_for_account
used to cap sizing at exactly 100% of buying_power, leaving no room for a
market order's real fill price to differ from the signal-time reference
price it was sized against.

    .venv/Scripts/python test_buying_power_buffer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide  # noqa: E402
from backtester.execution import BUYING_POWER_BUFFER, SizingMode, compute_qty_for_account  # noqa: E402


class FakeAccount(BrokerAccount):
    nickname = "FakeLive"
    is_paper = False

    def __init__(self, buying_power: float, equity: float | None = None):
        self._buying_power = buying_power
        self._equity = equity if equity is not None else buying_power

    def get_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            account_id="fake", equity=self._equity, cash=self._buying_power,
            buying_power=self._buying_power, is_paper=False,
        )

    def get_positions(self):
        return []

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        return []

    def submit_market_order(self, ticker: str, side: OrderSide, qty: float,
                             take_profit_price: float | None = None,
                             stop_loss_price: float | None = None) -> OrderResult:
        raise NotImplementedError


def test_sizing_leaves_a_buffer_below_full_buying_power():
    # Reproduces the real failure: $24.12 free, a ticker sized to spend it
    # ALL would round to a cost_basis that a real fill (at a slightly
    # different price than the signal-time reference) can tip over the
    # actual limit. The fix must leave real headroom, not spend it to zero.
    account = FakeAccount(buying_power=24.12)
    qty = compute_qty_for_account(
        account, reference_price=58.35, sizing_mode=SizingMode.PCT_EQUITY,
        sizing_value=100.0, ticker="DGX",
    )
    dollars_spent = qty * 58.35
    assert dollars_spent <= account._buying_power * BUYING_POWER_BUFFER + 1e-9
    assert dollars_spent < account._buying_power, (
        "sizing spent the account's ENTIRE buying power with no buffer - "
        "this is exactly the bug that caused repeated live rejections"
    )


def test_buffer_is_the_expected_98_percent():
    assert BUYING_POWER_BUFFER == 0.98


def test_small_price_uptick_after_sizing_still_fits_within_buying_power():
    # The real-world failure mode: reference_price at signal time, a
    # slightly higher price at actual fill time. With the buffer, the
    # resulting cost at the HIGHER price must still fit under buying_power.
    account = FakeAccount(buying_power=24.12)
    signal_price = 58.35
    qty = compute_qty_for_account(
        account, reference_price=signal_price, sizing_mode=SizingMode.PCT_EQUITY,
        sizing_value=100.0, ticker="DGX",
    )
    fill_price = signal_price * 1.01  # a modest 1% tick between signal and fill
    cost_at_fill = qty * fill_price
    assert cost_at_fill <= account._buying_power, (
        f"cost at fill (${cost_at_fill:.2f}) still exceeds buying power "
        f"(${account._buying_power:.2f}) even with the buffer - fix didn't work"
    )


if __name__ == "__main__":
    test_sizing_leaves_a_buffer_below_full_buying_power()
    test_buffer_is_the_expected_98_percent()
    test_small_price_uptick_after_sizing_still_fits_within_buying_power()
    print("All tests passed.")
