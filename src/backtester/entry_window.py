"""No NEW equity entries in the first minutes after the US open (added 2026-11-17).

WHY. The strategy lab (chopper-research-agent, Study I and its one-shot
hold-out Study I-H, docs/ROSTER_RISK_IDEAS_PLAN.md) replayed the roster's
regular-hours trades and found that entries made 09:30-09:44 New York lose more
than later ones: pooled over five walk-forward folds the gap is -5.9 bps a trade
(interval -11.9 to -0.0035) and the value over skipping the same share of
trades at random is +6.7 bps (+0.20 to +13.5). The pre-registered rule passed,
NARROWLY. This is a small cost trim on a book that loses about 10 bps a trade
either way - it does NOT make any pair profitable - and it is being run live
on Chris's decision to see what happens, not because it is proven.

WHAT. Only OPENING an equity position is held back; a close is never blocked
(de-risking is never blocked, same principle as the storm regime, the event-day
block and the account breaker). Crypto and forex are 24/7 / not on the US
session and are untouched. So is the index (IG) Opening Spike Fade strategy,
whose whole job is the first 25 minutes: it trades an index epic, which
`infer_asset_class` calls "forex" (".MINI."/".CFD."), never "equity".

Setting: `AutoTraderControl.skip_opening_minutes` (default 15, 0 = off). The
window is [09:30, 09:30 + N) America/New_York on a weekday, by the wall clock
at the moment the order would be placed (the lab classified by the entry bar's
time, and a live market order fills in the minute it is sent).
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

_NY_TZ = ZoneInfo("America/New_York")
_OPEN_MINUTES = 9 * 60 + 30  # 09:30 America/New_York
MAX_SKIP_MINUTES = 60  # a typo guard: nobody means "skip the first two hours" by accident


def in_opening_window(now_utc: datetime | None, skip_minutes: int) -> bool:
    """True when `now_utc` falls in [09:30, 09:30 + skip_minutes) New York time on a weekday."""
    if not isinstance(skip_minutes, int) or isinstance(skip_minutes, bool):
        return False
    if skip_minutes <= 0:
        return False
    skip_minutes = min(skip_minutes, MAX_SKIP_MINUTES)
    now_ny = (now_utc or datetime.now(timezone.utc)).astimezone(_NY_TZ)
    if now_ny.weekday() >= 5:
        return False
    minutes_now = now_ny.hour * 60 + now_ny.minute
    return _OPEN_MINUTES <= minutes_now < _OPEN_MINUTES + skip_minutes


def opening_block_reason(
    asset_class: str, is_opening: bool, now_utc: datetime | None, skip_minutes: int
) -> str | None:
    """A short reason string when this order must be held back, else None.

    Held back only when ALL of: it OPENS a position, the asset class is equity,
    and the time is inside the window. Anything else returns None.
    """
    if not is_opening or asset_class != "equity":
        return None
    if not in_opening_window(now_utc, skip_minutes):
        return None
    return f"opening-minutes rule (no new equity entries in the first {min(skip_minutes, MAX_SKIP_MINUTES)} min after 09:30 New York)"
