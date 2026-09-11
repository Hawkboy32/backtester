"""Regression check (2026-09-07): CoinbaseBroker.get_live_bars() must always
fetch USD-quoted candles, even when the account's own quote_currency is
something else (GBP for CBAPI) - it's a market-data method whose contract is
"shaped identically to Polygon's bars" for a Polygon-style ticker, not an
execution method. Only native_reference_price/submit_market_order should
follow the account's real trading currency.

THE BUG THIS FIXES (found live 2026-09-07, via the mobile app's 5m/15m
chart showing BTC at ~$58.7k - its real GBP price - while the 1m view,
sourced from a cached KRKAPI snapshot, correctly showed ~$79.4k):
get_live_bars used to call self._product_id(ticker) with no override,
which defaults to self.quote_currency - so for CBAPI (GBP) it silently
fetched BTC-GBP candles and returned them mislabeled as X:BTCUSD.
"""
from backtester.accounts import build_broker_accounts

CBAPI_ID = "f83b0c71-85db-47f9-ab69-9e51fb44bd19"
acct = build_broker_accounts([CBAPI_ID])[0]

assert acct.quote_currency == "GBP", "test assumes CBAPI is still GBP-configured"

# Market-data product id must be USD regardless of the account's own currency.
assert acct._product_id("X:BTCUSD", quote_currency="USD") == "BTC-USD"
# Execution/sizing product id must still be the account's real currency.
assert acct._product_id("X:BTCUSD") == "BTC-GBP"

from datetime import date, timedelta
to_date = date.today().isoformat()
from_date = (date.today() - timedelta(days=1)).isoformat()
df = acct.get_live_bars("X:BTCUSD", from_date, to_date, multiplier=15, timespan="minute")
assert not df.empty, "expected real live bars back"
last_close = float(df["close"].iloc[-1])
# A GBP-mislabeled bug would put this around ~55k-62k (BTC/GBP); the real
# USD price today is in the high $70ks-low $80ks. A loose sanity band, not
# a tight price assertion (this is live market data, not a fixture).
assert last_close > 65000, (
    f"last close {last_close} looks GBP-scaled, not USD - get_live_bars regressed"
)

print(f"PASS: get_live_bars returns USD-scaled data (last close ${last_close:,.2f}), "
      f"execution product_id still correctly BTC-GBP")
