"""Regression test: Alpaca orders record WHEN they were submitted and filled, and the quote just after
submission, so trading cost can be measured instead of estimated (2026-10-31).

Why: AlpacaLive's realized trades (179, Aug 11 - Oct 30) were down $3.33, and matching its fills to
cached minute bars put the cost at a median 4.3 / mean 6.3 bps a side, against the 2 bps every backtest
assumes. But 47% of those fills fell outside their own minute bar's high-low range, because neither
the fill time nor the market at submission was recorded. This test pins the new recording AND the one
property that matters more than any of it: measuring must never turn a sent order into a reported
failure (the caller could retry it into a duplicate order) and never delay it.

No network: the Alpaca clients are fakes. Run:
    .venv\\Scripts\\python.exe test_alpaca_fill_quality_fields.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from alpaca.trading.enums import OrderStatus  # noqa: E402

from backtester import execution_log  # noqa: E402
from backtester.brokers import alpaca as alpaca_mod  # noqa: E402
from backtester.brokers.alpaca import AlpacaBroker  # noqa: E402
from backtester.brokers.base import OrderResult, OrderSide  # noqa: E402

PASS, FAIL = [], []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -  {detail}" if detail and not ok else ""))


SUBMITTED = datetime(2026, 10, 30, 14, 31, 5, 120000, tzinfo=timezone.utc)
FILLED = datetime(2026, 10, 30, 14, 31, 5, 480000, tzinfo=timezone.utc)
QUOTED = datetime(2026, 10, 30, 14, 31, 5, 200000, tzinfo=timezone.utc)


def order(status, filled_qty=None, price=None, submitted=SUBMITTED, filled=None):
    return SimpleNamespace(id="ord-1", status=status, filled_qty=filled_qty, filled_avg_price=price, submitted_at=submitted, filled_at=filled)


class FakeTrading:
    def __init__(self, first, later=(), log=None, submit_raises=None):
        self.first, self.later, self.log, self.submit_raises = first, list(later), log if log is not None else [], submit_raises

    def submit_order(self, order_data):
        self.log.append("submit")
        if self.submit_raises:
            raise self.submit_raises
        return self.first

    def get_order_by_id(self, order_id):
        self.log.append("poll")
        return self.later.pop(0) if self.later else self.first


class FakeData:
    def __init__(self, quote=None, raises=None, log=None, missing=False):
        self.quote, self.raises, self.log, self.missing = quote, raises, log if log is not None else [], missing
        self.requests = []

    def get_stock_latest_quote(self, request):
        self.log.append("quote")
        self.requests.append(request)
        if self.raises:
            raise self.raises
        return {} if self.missing else {"AAPL": self.quote}


def broker(trading, data):
    b = AlpacaBroker.__new__(AlpacaBroker)        # no real clients, no keys, no network
    b.nickname, b.is_paper, b._client, b._data_client = "TestAlpaca", True, trading, data
    return b


def quote(bid=101.20, ask=101.24, ts=QUOTED):
    return SimpleNamespace(bid_price=bid, ask_price=ask, timestamp=ts)


def main() -> int:
    alpaca_mod.time.sleep = lambda s: None          # the fill poll's waits, made instant

    # --- the ordinary case: filled at once, quote available
    log = []
    r = broker(FakeTrading(order(OrderStatus.FILLED, 0.25, 101.23, filled=FILLED), log=log), FakeData(quote(), log=log)).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("a filled order is still reported as a success with its real fill", r.success and r.filled_qty == 0.25 and r.filled_avg_price == 101.23, str(r))
    check("submitted_at is the broker's own submission time, ISO-8601", r.submitted_at == SUBMITTED.isoformat(), str(r.submitted_at))
    check("filled_at is the broker's own fill time", r.filled_at == FILLED.isoformat(), str(r.filled_at))
    check("the quote's bid, ask and time are recorded", (r.quote_bid, r.quote_ask, r.quote_at) == (101.20, 101.24, QUOTED.isoformat()), f"{r.quote_bid} {r.quote_ask} {r.quote_at}")
    check("the quote is fetched AFTER the order is sent (measuring never delays it)", log.index("quote") > log.index("submit"), str(log))

    # --- filled only after polling: the quote is taken once, before the poll (closest to submission), and filled_at comes from the FINAL order
    log = []
    first = order(OrderStatus.NEW)
    later = [order(OrderStatus.PARTIALLY_FILLED, 0.1, 101.22), order(OrderStatus.FILLED, 0.25, 101.25, filled=FILLED)]
    data = FakeData(quote(), log=log)
    r = broker(FakeTrading(first, later, log=log), data).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("a fill after polling still records the final quantity and price", r.filled_qty == 0.25 and r.filled_avg_price == 101.25, str(r))
    check("...and the fill time of the final, filled order", r.filled_at == FILLED.isoformat(), str(r.filled_at))
    check("the quote is fetched exactly once", log.count("quote") == 1, str(log))
    check("...between the submission and the first poll", log == ["submit", "quote", "poll", "poll"], str(log))
    check("the quote request asks for the IEX feed (the only one this account may read live)", getattr(data.requests[0], "feed", None) == alpaca_mod.DataFeed.IEX, str(data.requests[0]))

    # --- the property that matters most: a failing measurement never makes a sent order look failed
    for label, d in (("the quote call raises", FakeData(raises=RuntimeError("quote endpoint down"))),
                     ("the symbol is missing from the reply", FakeData(missing=True)),
                     ("the quote object has the wrong shape", FakeData(quote=SimpleNamespace()))):
        r = broker(FakeTrading(order(OrderStatus.FILLED, 0.25, 101.23, filled=FILLED)), d).submit_market_order("AAPL", OrderSide.BUY, 0.25)
        check(f"{label}: the order is STILL a success with its fill", r.success and r.filled_avg_price == 101.23 and r.error is None, str(r))
        check(f"{label}: the quote fields are simply empty", (r.quote_bid, r.quote_ask, r.quote_at) == (None, None, None), f"{r.quote_bid} {r.quote_ask} {r.quote_at}")
        check(f"{label}: the order's own times are still recorded", r.submitted_at == SUBMITTED.isoformat() and r.filled_at == FILLED.isoformat())

    # an odd timestamp object (no isoformat) must not break anything either
    odd = order(OrderStatus.FILLED, 0.25, 101.23, submitted=object(), filled=object())
    r = broker(FakeTrading(odd), FakeData(quote(ts=object()))).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("timestamps the SDK might return in another shape become None, and the order is still a success", r.success and r.submitted_at is None and r.filled_at is None and r.quote_at is None and r.quote_bid == 101.20, str(r))
    # an order that has not filled (market shut) keeps filled_at empty
    r = broker(FakeTrading(order(OrderStatus.ACCEPTED)), FakeData(quote())).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("an unfilled (queued) order has no fill time, and is still reported the same way as before", r.success and r.filled_at is None and r.filled_qty is None, str(r))
    # a missing side of the book is recorded as given (None), not invented
    r = broker(FakeTrading(order(OrderStatus.FILLED, 0.25, 101.23, filled=FILLED)), FakeData(quote(bid=None))).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("a quote with no bid records bid None and keeps the ask", r.quote_bid is None and r.quote_ask == 101.24, f"{r.quote_bid} {r.quote_ask}")
    # a genuine submission failure is reported exactly as before, and nothing is measured
    log = []
    data = FakeData(quote(), log=log)
    r = broker(FakeTrading(None, log=log, submit_raises=RuntimeError("insufficient buying power")), data).submit_market_order("AAPL", OrderSide.BUY, 0.25)
    check("a rejected submission is still a failure with the broker's message", (not r.success) and "insufficient buying power" in (r.error or ""), str(r))
    check("...and no quote was fetched for an order that was never sent", "quote" not in log, str(log))

    # --- compatibility of the shared dataclass
    old = OrderResult("acct", True, "id-9", 1.0, 2.0, None)
    check("an OrderResult built positionally the old way still works, with every new field None", old.broker_order_id == "id-9" and old.filled_avg_price == 2.0 and old.submitted_at is None and old.quote_ask is None)

    # --- the execution log carries the new fields with no change to execution_log.py
    with tempfile.TemporaryDirectory() as tmp:
        saved = execution_log.LOG_PATH
        execution_log.LOG_PATH = Path(tmp) / "execution_log.jsonl"
        try:
            r = broker(FakeTrading(order(OrderStatus.FILLED, 0.25, 101.23, filled=FILLED)), FakeData(quote())).submit_market_order("AAPL", OrderSide.BUY, 0.25)
            execution_log.log_results("AAPL", OrderSide.BUY, [r])
            row = json.loads(execution_log.LOG_PATH.read_text(encoding="utf-8").strip())
            check("the execution log line has submitted_at, filled_at, quote_bid, quote_ask and quote_at", all(k in row for k in ("submitted_at", "filled_at", "quote_bid", "quote_ask", "quote_at")) and row["quote_ask"] == 101.24, str(row))
            check("...alongside the fields it always had", row["filled_avg_price"] == 101.23 and row["side"] == "buy" and row["ticker"] == "AAPL")
            df = execution_log.load_log()
            check("load_log still reads it (the dashboard's view)", len(df) == 1 and df.iloc[0]["quote_bid"] == 101.20)
        finally:
            execution_log.LOG_PATH = saved

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
