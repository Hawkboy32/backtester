from __future__ import annotations

from types import SimpleNamespace

import risk_node
from backtester import account_risk, accounts, auto_trader_state, daily_pnl_guard


class _FakeBrokerAccount:
    def __init__(self, account_id, nickname, equity=1000.0, cash=500.0, is_paper=True, positions=None, raises=False):
        self.account_id = account_id
        self.nickname = nickname
        self._equity = equity
        self._cash = cash
        self._is_paper = is_paper
        self._positions = positions or []
        self._raises = raises

    def get_account_snapshot(self):
        if self._raises:
            raise ConnectionError("broker unreachable")
        return SimpleNamespace(equity=self._equity, cash=self._cash, is_paper=self._is_paper)

    def get_positions(self):
        if self._raises:
            raise ConnectionError("broker unreachable")
        return self._positions


_FAKE_CONTROL = SimpleNamespace(
    sizing_mode="pct_equity",
    sizing_value=20.0,
    max_drawdown_enabled=True,
    max_drawdown_pct=15.0,
    giveback_enabled=True,
    giveback_pct=5.0,
)


def test_gather_risk_reports_equity_cash_and_positions(monkeypatch):
    position = SimpleNamespace(ticker="AAPL", qty=10, side="long", market_value=1900.0, unrealized_pl=50.0)
    account = _FakeBrokerAccount("acct-1", "MyAlpaca", equity=5000.0, cash=3000.0, positions=[position])

    monkeypatch.setattr(auto_trader_state, "load_control", lambda: _FAKE_CONTROL)
    monkeypatch.setattr(accounts, "build_broker_accounts", lambda account_ids=None: [account])
    monkeypatch.setattr(account_risk, "get_status", lambda account_id: None)
    monkeypatch.setattr(daily_pnl_guard, "get_status", lambda account_id: None)

    result = risk_node.gather_risk()

    assert len(result["accounts"]) == 1
    entry = result["accounts"][0]
    assert entry["account_id"] == "acct-1"
    assert entry["nickname"] == "MyAlpaca"
    assert entry["equity"] == 5000.0
    assert entry["error"] is None
    assert entry["positions"] == [
        {"ticker": "AAPL", "qty": 10, "side": "long", "market_value": 1900.0, "unrealized_pl": 50.0}
    ]
    assert result["sizing_mode"] == "pct_equity"
    assert result["max_drawdown_pct"] == 15.0


def test_gather_risk_isolates_one_brokers_failure_from_the_others(monkeypatch):
    healthy = _FakeBrokerAccount("acct-ok", "Healthy")
    broken = _FakeBrokerAccount("acct-broken", "Broken", raises=True)

    monkeypatch.setattr(auto_trader_state, "load_control", lambda: _FAKE_CONTROL)
    monkeypatch.setattr(accounts, "build_broker_accounts", lambda account_ids=None: [healthy, broken])
    monkeypatch.setattr(account_risk, "get_status", lambda account_id: None)
    monkeypatch.setattr(daily_pnl_guard, "get_status", lambda account_id: None)

    result = risk_node.gather_risk()

    by_id = {e["account_id"]: e for e in result["accounts"]}
    assert by_id["acct-ok"]["error"] is None
    assert "unreachable" in by_id["acct-broken"]["error"]
    # the broken account's failure must not have kept the healthy one's data out
    assert by_id["acct-ok"]["equity"] == 1000.0


def test_gather_risk_surfaces_tripped_breakers(monkeypatch):
    account = _FakeBrokerAccount("acct-1", "MyAlpaca")
    monkeypatch.setattr(auto_trader_state, "load_control", lambda: _FAKE_CONTROL)
    monkeypatch.setattr(accounts, "build_broker_accounts", lambda account_ids=None: [account])
    monkeypatch.setattr(account_risk, "get_status", lambda account_id: {"blocked": True, "reason": "drawdown exceeded"})
    monkeypatch.setattr(daily_pnl_guard, "get_status", lambda account_id: None)

    result = risk_node.gather_risk()

    assert result["accounts"][0]["drawdown_breaker"] == {"blocked": True, "reason": "drawdown exceeded"}
