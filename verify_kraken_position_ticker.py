"""Regression test for the Kraken position-ticker translation fix (2026-09-06).

Verifies the actual bug this closes: get_positions() used to return Kraken's
raw balance-asset-code ("XXBT") as a Position's ticker, which never matched
this app's "X:BTCUSD" ticker anywhere auto_trader.py compares positions -
the bot always saw existing_position=None for a real, live BTC position, so
it kept re-buying on every signal and could never recognize a SELL as a
close. See kraken.py's _KRAKEN_BALANCE_KEY_TO_SYMBOL docstring for the
live-verified mapping this depends on.

Entirely offline - no real broker calls (KrakenBroker is constructed with
fake credentials and _private() is monkeypatched to return a canned Balance
response, exactly the shape krakenex's query_private returns).
Run: python verify_kraken_position_ticker.py
"""

from __future__ import annotations

from backtester.brokers.kraken import KrakenBroker

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


kr = KrakenBroker(nickname="test", api_key="x", api_secret="y")

# A realistic Balance response: some fiat (must be skipped), a legacy-
# prefixed asset (XXBT), a non-prefixed asset (ADA), a zero-balance row
# (must be skipped), and one asset NOT in the verified table (must fall
# back to the raw code rather than raising or dropping the row).
kr._private = lambda method, data=None: {
    "ZUSD": "12.34",       # fiat - must not appear as a position
    "XXBT": "0.0001",      # THE bug case - legacy-prefixed
    "ADA": "150.0",        # non-prefixed asset
    "XETH": "0.0",         # zero balance - must be skipped
    "SOMENEWCOIN": "5.0",  # outside the verified table - fallback path
}

positions = {p.ticker: p for p in kr.get_positions()}

check("fiat (ZUSD) does not appear as a position", "ZUSD" not in positions)
check("zero-balance row (XETH) is skipped", not any(p.qty == 0 for p in positions.values()))
check(
    "THE bug: XXBT translates to X:BTCUSD, not the raw 'XXBT'",
    "X:BTCUSD" in positions and "XXBT" not in positions,
)
check("qty is preserved through the translation", positions.get("X:BTCUSD") and positions["X:BTCUSD"].qty == 0.0001)
check("non-prefixed asset (ADA) also translates correctly", "X:ADAUSD" in positions)
check(
    "an asset outside the verified table falls back to its raw code, not dropped or raised",
    "SOMENEWCOIN" in positions,
)

# The actual real-world consequence: reproduce auto_trader.py's own
# existing_position lookup line-for-line against the fixed output.
ticker = "X:BTCUSD"
existing_position = next((p for p in positions.values() if p.ticker == ticker and p.qty > 0), None)
check(
    "auto_trader.py's exact existing_position lookup now finds the BTC position "
    "(this was the live consequence: it always came back None before the fix)",
    existing_position is not None,
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All checks passed.")
