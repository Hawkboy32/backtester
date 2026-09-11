"""Regression test for the Coinbase native-currency sizing fix (2026-09-05).

Verifies the actual bug this closes: an account holding GBP (no USD wallet at
all - CBAPI, confirmed live) needs to execute BTC-GBP instead of BTC-USD, but
compute_qty_for_account was dividing GBP equity by a USD reference price -
silently under-sizing every order by the GBP/USD rate, with no error. See
CoinbaseBroker.native_reference_price's own docstring and execution.py's
compute_qty_for_account for the fix.

Entirely offline - no real broker calls (Coinbase's REST client is never
constructed with real credentials; the one network call native_reference_price
would make is monkeypatched). Run: python verify_coinbase_native_currency.py
"""

from __future__ import annotations

from dataclasses import dataclass

from backtester.brokers.base import AccountSnapshot, BrokerAccount
from backtester.brokers.coinbase import CoinbaseBroker
from backtester.execution import SizingMode, compute_qty_for_account

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------
# 1. _product_id respects quote_currency - unchanged default, overridden
#    when configured.
# ---------------------------------------------------------------------
cb_usd = CoinbaseBroker(nickname="default", api_key="x", api_secret="y")
check("default quote_currency is USD (unchanged)", cb_usd.quote_currency == "USD")
check("default _product_id: X:BTCUSD -> BTC-USD", cb_usd._product_id("X:BTCUSD") == "BTC-USD")

cb_gbp = CoinbaseBroker(nickname="CBAPI", api_key="x", api_secret="y", quote_currency="GBP")
check("overridden _product_id: X:BTCUSD -> BTC-GBP", cb_gbp._product_id("X:BTCUSD") == "BTC-GBP")
check(
    "an already product-id-shaped ticker still passes through unchanged",
    cb_gbp._product_id("ETH-USD") == "ETH-USD",
)

# ---------------------------------------------------------------------
# 2. native_reference_price is a NO-OP for the default (USD) instance - no
#    network call, returns the fallback untouched. This is what guarantees
#    zero behavior change for every account that never opts in.
# ---------------------------------------------------------------------
check(
    "USD instance: native_reference_price is a pure no-op (no client touched)",
    cb_usd.native_reference_price("X:BTCUSD", 79702.78) == 79702.78,
)

# ---------------------------------------------------------------------
# 3. Every OTHER broker type inherits the base no-op untouched - proves this
#    change cannot affect equities/forex/Kraken sizing at all.
# ---------------------------------------------------------------------
@dataclass
class PlainBroker(BrokerAccount):
    nickname: str = "plain"
    is_paper: bool = False
    def get_account_snapshot(self): raise NotImplementedError
    def get_positions(self): raise NotImplementedError
    def get_equity_history(self, period="1M", timeframe="1D"): raise NotImplementedError
    def submit_market_order(self, ticker, side, qty, take_profit_price=None, stop_loss_price=None): raise NotImplementedError

plain = PlainBroker()
check(
    "a broker with no override (Alpaca/IBKR/OANDA/IG/Kraken shape) is unaffected",
    plain.native_reference_price("AAPL", 231.40) == 231.40,
)

# ---------------------------------------------------------------------
# 4. THE actual bug: compute_qty_for_account with a mocked GBP-native price
#    must size off the GBP price, not the USD signal price it was handed -
#    proving the currency mismatch is actually closed, not just plumbed.
# ---------------------------------------------------------------------
class FakeGbpCoinbase(CoinbaseBroker):
    """Same class under test, but native_reference_price is monkeypatched to
    avoid a real network call while still exercising compute_qty_for_account's
    call site exactly as it would be in production."""
    def __init__(self):
        self.nickname = "CBAPI"
        self.is_paper = False
        self.quote_currency = "GBP"
        self._native_price = 60000.0  # a plausible standalone BTC-GBP quote

    def native_reference_price(self, ticker, fallback_price):
        assert self.quote_currency != "USD"
        return self._native_price  # what a real get_product() call would return

    def get_account_snapshot(self):
        # A bigger balance than CBAPI's real one, deliberately - at real
        # pilot-account scale (~GBP12) the OTHER known bug (qty rounded to
        # 4dp) happens to flatten both the correct and buggy prices to the
        # same 0.0001 BTC, which would make this assertion pass by accident
        # regardless of whether the currency fix works. Scaling up here
        # isolates and proves THIS fix specifically.
        return AccountSnapshot(account_id="cb", equity=1200.0, cash=1200.0, buying_power=1200.0, is_paper=False)

usd_signal_price = 79702.78  # the Polygon X:BTCUSD close the strategy actually saw
gbp_account = FakeGbpCoinbase()

qty_fixed = compute_qty_for_account(gbp_account, usd_signal_price, SizingMode.PCT_EQUITY, 50.0, "X:BTCUSD")
correct_qty = round((1200.0 * 0.5) / 60000.0, 4)
buggy_qty_if_unfixed = round((1200.0 * 0.5) / usd_signal_price, 4)

check(
    f"sizes off the GBP price ({qty_fixed} BTC, expected {correct_qty})",
    qty_fixed == correct_qty,
)
check(
    f"does NOT size off the raw USD signal price (would have been {buggy_qty_if_unfixed} BTC - "
    f"{buggy_qty_if_unfixed / correct_qty * 100:.0f}% of the correct size)",
    qty_fixed != buggy_qty_if_unfixed,
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All checks passed.")
