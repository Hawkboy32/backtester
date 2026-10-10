from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import activity_node
from backtester import auto_trader_state, current_signals, live_trades


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def test_gather_activity_reports_status_fields(monkeypatch):
    monkeypatch.setattr(
        auto_trader_state,
        "load_status",
        lambda: SimpleNamespace(running=True, trades_today=3, last_signal="AAPL/VWAP: BUY @ 190.0", last_error=None),
    )
    monkeypatch.setattr(current_signals, "load_signals", lambda: {})
    monkeypatch.setattr(live_trades, "list_recent_trades", lambda limit=20: [])

    result = activity_node.gather_activity()

    assert result["running"] is True
    assert result["trades_today"] == 3
    assert result["last_signal"] == "AAPL/VWAP: BUY @ 190.0"
    assert result["last_error"] is None


def test_gather_activity_includes_every_ticker_signal(monkeypatch):
    monkeypatch.setattr(
        auto_trader_state, "load_status", lambda: SimpleNamespace(running=True, trades_today=0, last_signal=None, last_error=None)
    )
    monkeypatch.setattr(
        current_signals,
        "load_signals",
        lambda: {
            "AAPL": {"signal": "hold", "strategy_name": "VWAP", "conviction": 0.4},
            "TSLA": {"signal": "buy", "strategy_name": "LRC", "conviction": 0.8},
        },
    )
    monkeypatch.setattr(live_trades, "list_recent_trades", lambda limit=20: [])

    result = activity_node.gather_activity()

    tickers = {s["ticker"] for s in result["per_ticker_signals"]}
    assert tickers == {"AAPL", "TSLA"}


def test_gather_activity_filters_closed_trades_to_today_only(monkeypatch):
    monkeypatch.setattr(
        auto_trader_state, "load_status", lambda: SimpleNamespace(running=True, trades_today=1, last_signal=None, last_error=None)
    )
    monkeypatch.setattr(current_signals, "load_signals", lambda: {})
    monkeypatch.setattr(
        live_trades,
        "list_recent_trades",
        lambda limit=20: [
            {"ticker": "AAPL", "strategy_name": "VWAP", "pnl": 12.5, "pnl_pct": 1.1, "account_id": "a1", "exit_time": f"{_today()}T10:00:00+00:00"},
            {"ticker": "TSLA", "strategy_name": "LRC", "pnl": -4.0, "pnl_pct": -0.5, "account_id": "a1", "exit_time": "2020-01-01T10:00:00+00:00"},
        ],
    )

    result = activity_node.gather_activity()

    assert len(result["closed_trades_today"]) == 1
    assert result["closed_trades_today"][0]["ticker"] == "AAPL"


def test_gather_activity_with_nothing_going_on_still_returns_a_clean_shape(monkeypatch):
    monkeypatch.setattr(
        auto_trader_state, "load_status", lambda: SimpleNamespace(running=False, trades_today=0, last_signal=None, last_error=None)
    )
    monkeypatch.setattr(current_signals, "load_signals", lambda: {})
    monkeypatch.setattr(live_trades, "list_recent_trades", lambda limit=20: [])

    result = activity_node.gather_activity()

    assert result == {
        "running": False,
        "trades_today": 0,
        "last_signal": None,
        "last_error": None,
        "per_ticker_signals": [],
        "closed_trades_today": [],
    }
