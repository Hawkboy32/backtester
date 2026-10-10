"""Risk/exposure node: real live exposure across every linked account, via
real authenticated broker connections - the one node in this graph that
touches the outside world. Strictly read-only: only ever calls
get_account_snapshot()/get_positions(), nothing that places, modifies, or
cancels an order. One broker's connection failing must not take the whole
node down - each account's own fetch is wrapped so a bad connection shows
up as that account's own `error` field, not a crashed run.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester import account_risk, accounts, auto_trader_state, daily_pnl_guard  # noqa: E402


def gather_risk(account_ids: list[str] | None = None) -> dict:
    control = auto_trader_state.load_control()
    broker_accounts = accounts.build_broker_accounts(account_ids)

    per_account = []
    for broker_account in broker_accounts:
        account_id = broker_account.account_id
        entry: dict = {"account_id": account_id, "nickname": broker_account.nickname, "error": None}
        try:
            snapshot = broker_account.get_account_snapshot()
            positions = broker_account.get_positions()
            entry["equity"] = snapshot.equity
            entry["cash"] = snapshot.cash
            entry["is_paper"] = snapshot.is_paper
            entry["positions"] = [
                {
                    "ticker": p.ticker,
                    "qty": p.qty,
                    "side": p.side,
                    "market_value": p.market_value,
                    "unrealized_pl": p.unrealized_pl,
                }
                for p in positions
            ]
        except Exception as exc:  # noqa: BLE001 - one broker's failure must not take down the rest
            entry["error"] = str(exc)

        entry["drawdown_breaker"] = account_risk.get_status(account_id)
        entry["giveback_breaker"] = daily_pnl_guard.get_status(account_id)
        per_account.append(entry)

    return {
        "accounts": per_account,
        "sizing_mode": control.sizing_mode,
        "sizing_value": control.sizing_value,
        "max_drawdown_enabled": control.max_drawdown_enabled,
        "max_drawdown_pct": control.max_drawdown_pct,
        "giveback_enabled": control.giveback_enabled,
        "giveback_pct": control.giveback_pct,
    }
