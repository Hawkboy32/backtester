"""Known-risk-event calendar: a PROACTIVE step-out filter.

"Don't sell insurance during a flood warning" — instead of waiting for
volatility to show up in the data (the GARCH regime filter in volatility.py
is reactive, it needs realized vol to spike first), step out of the market
around events that are KNOWN in advance to move it. New entries are blocked
on event days; exits are always allowed (de-risking is never blocked, same
principle as the storm regime and the account-risk breaker).

Phase 1 covers FOMC meeting days only — dates verified against the Fed's
official calendar (federalreserve.gov/monetarypolicy/fomccalendars.htm) on
2026-07-25. Both days of each two-day meeting are blocked: the decision
lands ~2pm ET on day two, but day-one positioning drift is part of the
event, and blocking the pair is the conservative, cheap choice.

Per-ticker earnings dates are the obvious phase 2 — but Polygon's earnings
calendar is a paid Benzinga add-on this account's plan does not include
(verified 2026-07-25: HTTP 403 "not entitled" on /benzinga/v1/earnings).
The `ticker` parameter on is_event_day()/event_reason() is the seam where
an earnings provider plugs in later without touching any caller.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

# (first_day, last_day) of each scheduled FOMC meeting. Source: the Fed's
# official calendar page, fetched 2026-07-25. The 2025-08-22 "notation vote"
# is deliberately excluded — it isn't a scheduled meeting with a 2pm decision.
# MAINTENANCE: the Fed publishes each year's schedule well in advance — add
# the next year's meetings when they appear (see CALENDAR_COVERS_THROUGH).
FOMC_MEETINGS: list[tuple[date, date]] = [
    # 2024
    (date(2024, 1, 30), date(2024, 1, 31)),
    (date(2024, 3, 19), date(2024, 3, 20)),
    (date(2024, 4, 30), date(2024, 5, 1)),
    (date(2024, 6, 11), date(2024, 6, 12)),
    (date(2024, 7, 30), date(2024, 7, 31)),
    (date(2024, 9, 17), date(2024, 9, 18)),
    (date(2024, 11, 6), date(2024, 11, 7)),
    (date(2024, 12, 17), date(2024, 12, 18)),
    # 2025
    (date(2025, 1, 28), date(2025, 1, 29)),
    (date(2025, 3, 18), date(2025, 3, 19)),
    (date(2025, 5, 6), date(2025, 5, 7)),
    (date(2025, 6, 17), date(2025, 6, 18)),
    (date(2025, 7, 29), date(2025, 7, 30)),
    (date(2025, 9, 16), date(2025, 9, 17)),
    (date(2025, 10, 28), date(2025, 10, 29)),
    (date(2025, 12, 9), date(2025, 12, 10)),
    # 2026
    (date(2026, 1, 27), date(2026, 1, 28)),
    (date(2026, 3, 17), date(2026, 3, 18)),
    (date(2026, 4, 28), date(2026, 4, 29)),
    (date(2026, 6, 16), date(2026, 6, 17)),
    (date(2026, 7, 28), date(2026, 7, 29)),
    (date(2026, 9, 15), date(2026, 9, 16)),
    (date(2026, 10, 27), date(2026, 10, 28)),
    (date(2026, 12, 8), date(2026, 12, 9)),
    # 2027
    (date(2027, 1, 26), date(2027, 1, 27)),
    (date(2027, 3, 16), date(2027, 3, 17)),
    (date(2027, 4, 27), date(2027, 4, 28)),
    (date(2027, 6, 8), date(2027, 6, 9)),
    (date(2027, 7, 27), date(2027, 7, 28)),
    (date(2027, 9, 14), date(2027, 9, 15)),
    (date(2027, 10, 26), date(2027, 10, 27)),
    (date(2027, 12, 7), date(2027, 12, 8)),
]

# Last date the calendar has knowledge of. Past this, is_event_day() returning
# False means "no data", not "no event" — callers that run indefinitely (the
# auto-trader) should surface calendar_covers() going False instead of
# silently trading through unlisted meetings.
CALENDAR_COVERS_THROUGH: date = max(end for _, end in FOMC_MEETINGS)


@lru_cache(maxsize=1)
def fomc_dates() -> frozenset[date]:
    """Every individual calendar day belonging to a scheduled FOMC meeting."""
    days: set[date] = set()
    for start, end in FOMC_MEETINGS:
        d = start
        while d <= end:
            days.add(d)
            d += timedelta(days=1)
    return frozenset(days)


def calendar_covers(d: date) -> bool:
    """Whether the calendar has knowledge of this date. False for dates after
    the last listed meeting — treat a False is_event_day() there as unknown."""
    return d <= CALENDAR_COVERS_THROUGH


def is_event_day(d: date, ticker: str | None = None) -> bool:
    """True if `d` is a known risk-event day. `ticker` is unused in phase 1
    (FOMC is market-wide); it's the seam for per-ticker earnings dates later.
    """
    return d in fomc_dates()


def event_reason(d: date, ticker: str | None = None) -> str | None:
    """Human-readable label for the event on `d`, or None."""
    if d in fomc_dates():
        return "FOMC meeting day"
    return None


def blocked_dates_in_range(from_date: date, to_date: date) -> set[date]:
    """All event days within [from_date, to_date] — for the backtest engine,
    which wants a plain set to check per bar."""
    return {d for d in fomc_dates() if from_date <= d <= to_date}
