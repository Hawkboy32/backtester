"""Regression test for the extra_targets account_ids routing fix
(2026-09-01). Verifies the actual bug found live on 2026-08-25 is closed:
an extra_targets group used to trade on EVERY account matching its ticker's
asset class, ignoring its own account_ids entirely. See auto_trader.py's
_resolve_targets docstring and CLAUDE_NOTES.txt "OPEN BUG 2026-08-25" for
the full incident.

Entirely offline - no broker calls, no real control.json/roster.json
touched. Run: python verify_target_routing.py
"""

from __future__ import annotations

from dataclasses import dataclass

from auto_trader import _resolve_targets
from backtester.auto_trader_state import AutoTraderControl

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------
# 1. Manual-mode primary: no extra_targets - account_ids must be None
#    (unrestricted, same as before the fix).
# ---------------------------------------------------------------------
control = AutoTraderControl(
    use_roster=False,
    tickers=["AAPL", "MSFT"],
    strategy_name="Bollinger Mean Reversion",
    account_ids=["acct-primary"],
)
targets = _resolve_targets(control)
check("primary-only: 2 targets returned", len(targets) == 2)
check(
    "primary targets all have account_ids=None (unrestricted)",
    all(t[4] is None for t in targets),
)

# ---------------------------------------------------------------------
# 2. Extra targets carry their OWN group's account_ids, not each
#    other's, and not the primary's. This is the actual bug: before the
#    fix, _resolve_targets didn't return this field at all.
# ---------------------------------------------------------------------
control = AutoTraderControl(
    use_roster=False,
    tickers=["AAPL"],
    strategy_name="Bollinger Mean Reversion",
    account_ids=["acct-primary"],
    extra_targets=[
        {"label": "EURUSD (IG paper)", "account_ids": ["acct-ig-paper"],
         "tickers": ["C:EURUSD"], "strategy_name": "Bollinger Mean Reversion", "strategy_params": {}},
        {"label": "4 pairs (OANDA)", "account_ids": ["acct-oanda"],
         "tickers": ["C:GBPUSD", "C:USDJPY"], "strategy_name": "Bollinger Mean Reversion", "strategy_params": {}},
    ],
)
targets = _resolve_targets(control)
by_ticker = {t[0]: t for t in targets}

check("AAPL (primary) is unrestricted", by_ticker["AAPL"][4] is None)
check(
    "C:EURUSD is scoped to ONLY acct-ig-paper",
    by_ticker["C:EURUSD"][4] == ["acct-ig-paper"],
)
check(
    "C:GBPUSD is scoped to ONLY acct-oanda, NOT acct-ig-paper",
    by_ticker["C:GBPUSD"][4] == ["acct-oanda"],
)
check(
    "C:USDJPY (same group as GBPUSD) also scoped to acct-oanda only",
    by_ticker["C:USDJPY"][4] == ["acct-oanda"],
)

# ---------------------------------------------------------------------
# 3. The actual scoping filter used in run_cycle's loop - reproduced
#    here exactly as it appears in auto_trader.py, against a set of fake
#    broker-account-like objects, to prove cross-contamination is gone.
# ---------------------------------------------------------------------
@dataclass
class FakeAccount:
    account_id: str


broker_accounts = [
    FakeAccount("acct-primary"),
    FakeAccount("acct-ig-paper"),
    FakeAccount("acct-oanda"),
]


def scope(target_account_ids, broker_accounts):
    return (
        broker_accounts if target_account_ids is None
        else [a for a in broker_accounts if a.account_id in target_account_ids]
    )


eurusd_scoped = scope(by_ticker["C:EURUSD"][4], broker_accounts)
check(
    "EURUSD's scoped account list is exactly [acct-ig-paper] - "
    "THIS is the fix: before it, EURUSD would have traded on ALL THREE",
    [a.account_id for a in eurusd_scoped] == ["acct-ig-paper"],
)

gbpusd_scoped = scope(by_ticker["C:GBPUSD"][4], broker_accounts)
check(
    "GBPUSD's scoped account list is exactly [acct-oanda], "
    "acct-ig-paper correctly excluded",
    [a.account_id for a in gbpusd_scoped] == ["acct-oanda"],
)

aapl_scoped = scope(by_ticker["AAPL"][4], broker_accounts)
check(
    "AAPL (primary, account_ids=None) still gets the FULL pool - "
    "confirms primary/manual/roster behavior is unchanged",
    len(aapl_scoped) == 3,
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All checks passed.")
