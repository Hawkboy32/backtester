"""Regression test for the Kraken zero-entry-price bug (found 2026-09-22
reviewing live-account P&L): kraken.py's get_positions() deliberately always
reports avg_entry_price=0.0 (Kraken's own API has no cost-basis field, see
that module's docstring) - but auto_trader.py's closing code used to trust
that broker-reported value directly when recording a realised trade, so
entry_price=0 got recorded and pnl was computed as almost the full exit
notional. 14 of 15 real KRKAPI round trips were wrong this way, overstating
live P&L by ~$109 against a true ~$0.01.

Fix: capture the REAL fill price at open time in position_attribution's own
record, and prefer that over whatever the broker reports at close time via
resolve_entry_price().

    .venv/Scripts/python test_kraken_entry_price_attribution.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester import position_attribution  # noqa: E402

# Redirect state to a throwaway temp dir so this test never touches the
# project's real auto_trader_state/positions.json.
_tmp_state = Path(tempfile.mkdtemp())
position_attribution.STATE_DIR = _tmp_state
position_attribution.PATH = _tmp_state / "positions.json"


def test_kraken_style_zero_entry_price_no_longer_wins():
    # Simulates exactly the real incident: Kraken always reports
    # avg_entry_price=0.0 on get_positions(), but the REAL fill price
    # (79706.8) was captured at open time.
    position_attribution.record_open(
        "kraken-account", "X:BTCUSD", "VWAP Mean Reversion", entry_price=79706.8,
    )
    attribution = position_attribution.pop_open("kraken-account", "X:BTCUSD")
    assert attribution is not None

    broker_reported_entry_price = 0.0  # what Kraken's get_positions() always returns
    resolved = position_attribution.resolve_entry_price(attribution, broker_reported_entry_price)

    assert resolved == 79706.8, f"got {resolved} - fell back to the broker's bogus 0.0, bug reintroduced"

    # Prove the P&L this actually fixes: a real, small, sane move - not the
    # near-full-notional "profit" the old bug produced.
    exit_price = 79764.3
    qty = 0.0001
    pnl = (exit_price - resolved) * qty
    assert abs(pnl - 0.00573) < 1e-4, f"pnl={pnl} - expected a tiny realistic move, not the notional-sized bug"


def test_falls_back_to_broker_value_when_no_entry_price_was_recorded():
    # A position opened before this fix existed (or whose attribution was
    # lost) has no "entry_price" key at all - must fall back honestly to
    # whatever the broker reports, not crash or silently drop the trade.
    attribution = {"strategy_name": "VWAP Mean Reversion", "opened_at": "", "conviction": None}
    resolved = position_attribution.resolve_entry_price(attribution, 142.59)
    assert resolved == 142.59


def test_normal_broker_with_a_real_avg_entry_price_still_prefers_recorded_fill():
    # Even for a broker that DOES report a real avg_entry_price (Alpaca),
    # the recorded open-time fill is still the source of truth - it's the
    # exact price actually paid, not a value the broker may have rounded
    # or blended differently.
    position_attribution.record_open("alpaca-account", "AAPL", "Bollinger Mean Reversion", entry_price=232.735306)
    attribution = position_attribution.pop_open("alpaca-account", "AAPL")
    resolved = position_attribution.resolve_entry_price(attribution, broker_reported_entry_price=232.70)
    assert resolved == 232.735306


if __name__ == "__main__":
    test_kraken_style_zero_entry_price_no_longer_wins()
    test_falls_back_to_broker_value_when_no_entry_price_was_recorded()
    test_normal_broker_with_a_real_avg_entry_price_still_prefers_recorded_fill()
    print("All tests passed.")
