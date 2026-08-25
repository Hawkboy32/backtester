"""How far back can we actually backtest? Paced probe of the Polygon plan.

Answers the question the multi-year validation plan depends on, and which a
fast probe gets WRONG: a burst of requests returns 429 (rate limited), which
looks identical to "no data" if you don't distinguish them. 403 is the real
signal — that is the plan's history ceiling.

Deliberately slow (the account's limit is a few requests/minute), so run it in
the background and read the result.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
load_dotenv()

from backtester.data import PolygonClient  # noqa: E402

TICKER = "BEN"          # long-listed, so absence of data means the PLAN, not the ticker
SPACING_SECONDS = 15    # well inside the documented 5/min

PROBES = [
    ("2021-06", "day"), ("2022-06", "day"), ("2023-06", "day"),
    ("2024-06", "day"), ("2025-06", "day"),
    ("2023-06", "minute"), ("2024-02", "minute"), ("2024-06", "minute"),
    ("2024-10", "minute"), ("2025-01", "minute"), ("2025-06", "minute"),
]


def main() -> int:
    client = PolygonClient(use_cache=True)
    print(f"probing {TICKER}, {SPACING_SECONDS}s apart\n")
    print(f"  {'window':<10}{'bars':>8}  result")
    for month, timespan in PROBES:
        year, mon = month.split("-")
        last = "28"
        try:
            bars = client.get_aggregates(TICKER, f"{month}-01", f"{month}-{last}", 1, timespan)
            print(f"  {month} {timespan:<7}{len(bars):>8}  ok")
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "403" in msg:
                verdict = "FORBIDDEN — beyond the plan's history ceiling"
            elif "429" in msg:
                verdict = "rate limited (inconclusive, not an availability answer)"
            else:
                verdict = msg[:70]
            print(f"  {month} {timespan:<7}{'-':>8}  {verdict}")
        time.sleep(SPACING_SECONDS)
    print("\n403 = the plan cannot go back further. 429 = probe too fast, says nothing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
