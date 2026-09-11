"""Regression check for the Kraken get_positions() live-price enrichment
(2026-09-07): current_price/market_value should now be populated for
recognized assets, using a real query against the live KRKAPI account.
"""
from backtester.accounts import build_broker_accounts

accounts = build_broker_accounts(["07623a39-a44c-4d65-b48f-178703308183"])
acct = accounts[0]
positions = acct.get_positions()

assert positions, "expected at least one open Kraken position (BTC dust from live testing)"
for p in positions:
    print(p)
    assert p.current_price is not None, f"{p.ticker}: current_price still None - enrichment didn't fire"
    assert p.current_price > 0, f"{p.ticker}: current_price not positive"
    assert p.market_value == p.qty * p.current_price, f"{p.ticker}: market_value doesn't match qty*current_price"

print("PASS: live price + market_value now populated for Kraken positions")

# --- Synthetic: prove a Ticker lookup failure can't regress the earlier
# position-recognition fix (2026-09-06) - the position must still come back
# with the right ticker/qty even if price enrichment throws.
from unittest.mock import patch

with patch.object(acct, "_ticker_price", side_effect=RuntimeError("simulated Kraken outage")):
    fallback_positions = acct.get_positions()

assert fallback_positions, "position vanished when price lookup failed - regression!"
p = fallback_positions[0]
assert p.ticker == "X:BTCUSD", f"ticker recognition broke under price-lookup failure: {p.ticker}"
assert p.qty > 0
assert p.current_price is None and p.market_value == 0.0, "should fall back cleanly, not half-fill"
print("PASS: price-lookup failure degrades gracefully, doesn't regress position recognition")
