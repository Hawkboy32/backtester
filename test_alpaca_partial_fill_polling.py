"""Regression test for the AlpacaBroker.submit_market_order partial-fill bug
(found 2026-09-22 while closing all 11 MyAlpaca positions): the fill-poll
loop used to stop as soon as filled_qty was ANY positive number, so a large
fractional-share order that fills across several partial lots got reported
as "done" after just the first lot - e.g. BAC requested qty=35.1167, first
partial landed at qty=16.0, and the function returned that as final even
though the order kept filling afterward. The position itself wasn't left
half-closed (Alpaca kept filling it), but any caller trusting the returned
filled_qty/filled_avg_price - realised P&L recording, copy-trade fill
tracking - would have recorded the wrong quantity and only the first lot's
price. Fixed to wait for order.status == FILLED specifically.

    .venv/Scripts/python test_alpaca_partial_fill_polling.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alpaca.trading.enums import OrderStatus  # noqa: E402

from backtester.brokers.alpaca import AlpacaBroker  # noqa: E402
from backtester.brokers.base import OrderSide  # noqa: E402


def _fake_broker(order_sequence: list[SimpleNamespace]) -> AlpacaBroker:
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker.nickname = "TestAccount"
    broker.is_paper = True
    broker._client = MagicMock()
    broker._client.submit_order = MagicMock(return_value=order_sequence[0])
    # get_order_by_id is called once per poll after the initial submit
    # response - return each subsequent state in sequence.
    broker._client.get_order_by_id = MagicMock(side_effect=order_sequence[1:])
    return broker


def _order(status, filled_qty, filled_avg_price, order_id="ord-1"):
    return SimpleNamespace(id=order_id, status=status, filled_qty=filled_qty, filled_avg_price=filled_avg_price)


def test_stops_polling_early_only_once_fully_filled():
    # Requested 35.1167 - fills in two partial lots, then fully filled.
    # Must NOT stop at the first partial (qty=16.0) like the old bug did.
    sequence = [
        _order(OrderStatus.NEW, "0", None),
        _order(OrderStatus.PARTIALLY_FILLED, "16", "56.57"),
        _order(OrderStatus.PARTIALLY_FILLED, "28.5", "56.55"),
        _order(OrderStatus.FILLED, "35.1167", "56.56"),
    ]
    broker = _fake_broker(sequence)

    result = broker.submit_market_order("BAC", OrderSide.SELL, 35.1167)

    assert result.success is True
    assert result.filled_qty == 35.1167, f"got {result.filled_qty} - stopped at a partial fill, bug reintroduced"
    assert result.filled_avg_price == 56.56


def test_gives_up_honestly_on_a_terminal_non_fill_status():
    # A cancelled order should NOT be polled forever - and the result
    # should reflect the true (zero/partial) state, not fabricate FILLED.
    sequence = [
        _order(OrderStatus.NEW, "0", None),
        _order(OrderStatus.CANCELED, "0", None),
    ]
    broker = _fake_broker(sequence)

    result = broker.submit_market_order("XYZ", OrderSide.BUY, 10)

    assert result.success is True  # order was accepted by the broker, just didn't fill
    assert result.filled_qty is None or result.filled_qty == 0.0


def test_still_returns_promptly_on_a_normal_single_shot_fill():
    # The common case (small order, fills immediately) shouldn't regress
    # into always exhausting the poll budget.
    sequence = [
        _order(OrderStatus.NEW, "0", None),
        _order(OrderStatus.FILLED, "1.0", "744.01"),
    ]
    broker = _fake_broker(sequence)

    result = broker.submit_market_order("META", OrderSide.SELL, 1.0)

    assert result.filled_qty == 1.0
    assert result.filled_avg_price == 744.01
    # only one get_order_by_id call needed, not the full poll budget
    assert broker._client.get_order_by_id.call_count == 1


if __name__ == "__main__":
    test_stops_polling_early_only_once_fully_filled()
    test_gives_up_honestly_on_a_terminal_non_fill_status()
    test_still_returns_promptly_on_a_normal_single_shot_fill()
    print("All tests passed.")
