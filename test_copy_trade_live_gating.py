"""Regression test for copy_trade_executor's live-account gating (added
2026-09-23): only LIVE_COPY_USERNAMES may ever get AlpacaLive as a target -
every other investor must stay paper-only, no matter what. Getting this
wrong means real money on the wrong signal, so this is tested directly
rather than trusted by inspection alone.

    .venv/Scripts/python test_copy_trade_live_gating.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import copy_trade_executor as cte  # noqa: E402

FAKE_ACCOUNTS = [
    {"id": "paper-1", "nickname": "IBKR Paper", "broker": "ibkr", "is_paper": True,
     "conn_params": {}},
    {"id": "live-alpaca", "nickname": "AlpacaLive", "broker": "alpaca", "is_paper": False},
    {"id": "live-ibkr", "nickname": "IBKR Live", "broker": "ibkr", "is_paper": False,
     "conn_params": {"asset_class": "equity"}},
    {"id": "etoro-1", "nickname": "ETPaper", "broker": "etoro", "is_paper": True},
]


def _run(username: str, ticker: str = "MU") -> list[str]:
    with patch.object(cte, "list_accounts", return_value=FAKE_ACCOUNTS), \
         patch.object(cte, "account_asset_class", return_value="equity"), \
         patch.object(cte, "infer_asset_class", return_value="equity"):
        targets = cte._target_accounts(username, ticker)
    return sorted(a["nickname"] for a in targets)


def test_non_live_eligible_investor_never_gets_a_live_account():
    for username in ("campervans", "celesh", "SomeRandomInvestor"):
        targets = _run(username)
        assert "AlpacaLive" not in targets, f"{username} got a LIVE account — this must never happen"
        assert "IBKR Live" not in targets
        assert targets == ["IBKR Paper"], f"{username}: expected only the paper account, got {targets}"


def test_live_eligible_investors_get_alpacalive_plus_paper():
    for username in cte.LIVE_COPY_USERNAMES:
        targets = _run(username)
        assert "AlpacaLive" in targets, f"{username} should be live-eligible but didn't get AlpacaLive"
        assert "IBKR Paper" in targets, f"{username} lost paper targeting — live should ADD, not replace"
        assert "IBKR Live" not in targets, "only the named live account should ever be targeted, not any live account"
        assert "ETPaper" not in targets, "eToro accounts must never be a copy-trade target"


def test_exactly_the_expected_two_usernames_are_live_eligible():
    # Locks in the actual decision (2026-09-23) — a silent change to this
    # set later should fail loudly, not drift unnoticed.
    assert cte.LIVE_COPY_USERNAMES == {"RainbirdFx", "Aukie2008"}


def test_live_target_account_nickname_is_alpacalive():
    assert cte.LIVE_COPY_ACCOUNT_NICKNAME == "AlpacaLive"


if __name__ == "__main__":
    test_non_live_eligible_investor_never_gets_a_live_account()
    test_live_eligible_investors_get_alpacalive_plus_paper()
    test_exactly_the_expected_two_usernames_are_live_eligible()
    test_live_target_account_nickname_is_alpacalive()
    print("All tests passed.")
