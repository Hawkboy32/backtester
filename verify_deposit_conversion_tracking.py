"""Regression test for the deposit-vs-real-conversion tracking added 2026-09-06.

Verifies: total_deposited() prefers a recorded real conversion over the
logged estimate, total_deposited_estimated() keeps the original numbers
visible regardless, and record_conversion() splits one lump conversion
across multiple still-pending deposits proportionally to their own amounts -
the realistic shape (three small GBP deposits, one bulk conversion).

Entirely offline - redirects deposits.py's own DEPOSITS_PATH to a temp file
before writing anything, never touches the real auto_trader_state/deposits.json.
Run: python verify_deposit_conversion_tracking.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from backtester import deposits

deposits.DEPOSITS_PATH = Path(tempfile.mkdtemp()) / "deposits.json"

failures: list[str] = []


def check(label: str, condition: bool) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        failures.append(label)


ACCOUNT = "test-account"

deposits.record_deposit(ACCOUNT, 6.82, "2026-08-20")
deposits.record_deposit(ACCOUNT, 2.65, "2026-09-01")
deposits.record_deposit(ACCOUNT, 6.75, "2026-09-05")

check(
    "before any conversion: total_deposited == total_deposited_estimated == sum of logged amounts",
    deposits.total_deposited(ACCOUNT) == deposits.total_deposited_estimated(ACCOUNT) == 16.22,
)

deposits.record_conversion(ACCOUNT, 15.81, converted_at="2026-09-05T08:39:31+00:00")

check(
    "after conversion: total_deposited reflects the REAL banked amount",
    abs(deposits.total_deposited(ACCOUNT) - 15.81) < 0.01,
)
check(
    "total_deposited_estimated is UNCHANGED - the original trend data is preserved",
    deposits.total_deposited_estimated(ACCOUNT) == 16.22,
)

entries = deposits.deposits_for(ACCOUNT)
check("all 3 entries got a converted_amount (proportional split)", all(e.converted_amount is not None for e in entries))
check(
    "the split amounts still sum to the real converted total (no money lost/gained in rounding)",
    abs(sum(e.converted_amount for e in entries) - 15.81) < 0.02,
)
check(
    "the largest original deposit ($6.82) got the largest converted share",
    max(entries, key=lambda e: e.amount).converted_amount == max(e.converted_amount for e in entries),
)

# A second, later conversion (a NEW deposit arriving after this one) must
# only touch the new pending entry, never re-touch the already-converted ones.
deposits.record_deposit(ACCOUNT, 3.00, "2026-09-10")
before = {e.recorded_at: e.converted_amount for e in deposits.deposits_for(ACCOUNT) if e.converted_amount is not None}
deposits.record_conversion(ACCOUNT, 4.05, converted_at="2026-09-10T09:00:00+00:00")
after = {e.recorded_at: e.converted_amount for e in deposits.deposits_for(ACCOUNT) if e.recorded_at in before}
check("a later conversion never re-touches already-converted entries", before == after)

try:
    deposits.record_conversion(ACCOUNT, 99.0)
    check("record_conversion raises when nothing is left pending", False)
except ValueError:
    check("record_conversion raises when nothing is left pending", True)

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    raise SystemExit(1)
print("All checks passed.")
