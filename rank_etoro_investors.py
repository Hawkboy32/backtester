"""Rank eToro Popular Investors on multiple risk-adjusted factors and print a
top-N shortlist — the selection step before copy_trader.py watches anyone.
Personal system only, not part of ChopperCommercial.

Sorting by raw gain alone is a real trap here, not a hypothetical one:
pulling live data 2026-09-16 sorted by -gain, the #1 result (pontix4) was a
345% gain crypto trader with riskScore=8, weeklyDD -20.6%, peakToValley
-41.8%, only 46% winRatio, and 0 copiers — exactly the "looked amazing on
one number, terrifying on every other" profile. Mirrors why this project's
own strategy ranking (ranking.py's DEFAULT_WEIGHTS) already weights a
Sharpe-like measure at 40% and raw return at only 30%: the same philosophy
applied to investors instead of strategies.

Composite score (min-max normalized across the candidate pool, same
_minmax pattern ranking.py uses):
  - risk_adjusted_return (0.35): annualizedReturn / riskScore — a crude
    "return per unit of the investor's OWN stated risk score" (1-10),
    closer to a Sharpe proxy than annualizedReturn alone.
  - drawdown_control (0.25): peakToValley (their worst verified drawdown),
    less negative = better.
  - consistency (0.20): mean of winRatio and profitableMonthsPct.
  - total_return (0.10): annualizedReturn alone — smaller weight since the
    risk-adjusted component above already captures most of what matters.
  - track_record (0.10): weeksSinceRegistration and trade count, both
    log-scaled — rewards a longer, more heavily-traded verified history
    over someone who looks great on 15 trades and two months.

Pre-filters (before scoring, not part of the score itself): popularInvestor
only (real people who've opted into being copied, not a smart-portfolio
index), riskScore <= 7 (excludes the most reckless tier), trades >= 20,
weeksSinceRegistration >= 26 (~6 months of real track record).

Usage:
    python rank_etoro_investors.py --top 5
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path

import keyring
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from backtester.accounts import KEYRING_SERVICE, _keyring_key, _load_raw  # noqa: E402

RANKINGS_URL = "https://public-api.etoro.com/api/v2/portfolios/rankings"

# Pull several sort orders so the candidate pool isn't just "whoever sorts
# top by one metric" — a real breadth of styles rather than one lens.
SORT_ORDERS = ["-gain", "-copiers", "-winRatio", "-annualizedReturn"]
PAGES_PER_SORT = 2
PAGE_SIZE = 50

MIN_TRADES = 20
MIN_WEEKS_SINCE_REGISTRATION = 26
MAX_RISK_SCORE = 7


def _headers(api_key: str, user_key: str) -> dict:
    return {
        "x-request-id": str(uuid.uuid4()),
        "x-api-key": api_key,
        "x-user-key": user_key,
        "Content-Type": "application/json",
    }


def _find_etoro_credentials(account_nickname: str | None) -> tuple[str, str]:
    candidates = [a for a in _load_raw() if a["broker"] == "etoro"]
    if account_nickname:
        candidates = [a for a in candidates if a["nickname"] == account_nickname]
    if not candidates:
        raise SystemExit("No linked eToro account found. Link one via the Accounts tab first.")
    account = candidates[0]
    api_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "api_key"))
    user_key = keyring.get_password(KEYRING_SERVICE, _keyring_key(account["id"], "secret_key"))
    if not api_key or not user_key:
        raise SystemExit(f"Credentials for '{account['nickname']}' are missing from the OS keyring.")
    return api_key, user_key


def fetch_candidates(api_key: str, user_key: str) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for sort in SORT_ORDERS:
        for page in range(1, PAGES_PER_SORT + 1):
            params = {
                "period": "LastYear",
                "page": page,
                "pageSize": PAGE_SIZE,
                "popularInvestor": "true",
                "riskScoreMax": MAX_RISK_SCORE,
                "sort": sort,
            }
            resp = requests.get(RANKINGS_URL, headers=_headers(api_key, user_key), params=params, timeout=20)
            resp.raise_for_status()
            for row in resp.json().get("results", []):
                rows[row["username"]] = row  # de-dupe across sorts/pages by username
    return pd.DataFrame(rows.values())


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def score_candidates(df: pd.DataFrame) -> pd.DataFrame:
    df = df[
        (df["trades"] >= MIN_TRADES) & (df["weeksSinceRegistration"] >= MIN_WEEKS_SINCE_REGISTRATION)
    ].copy()
    if df.empty:
        return df

    df["risk_adjusted_return"] = df["annualizedReturn"] / df["riskScore"].clip(lower=1)
    df["consistency"] = (df["winRatio"].fillna(0) + df["profitableMonthsPct"].fillna(0)) / 2
    # log1p so an investor with 500 trades / 5 years doesn't totally dwarf
    # one with a solid-but-shorter 40 trades / 8 months — diminishing
    # returns on "more history", not a straight linear reward for it.
    df["track_record"] = df["trades"].apply(lambda t: math.log1p(t)) + df["weeksSinceRegistration"].apply(
        lambda w: math.log1p(w)
    )

    df["score"] = (
        0.35 * _minmax(df["risk_adjusted_return"])
        + 0.25 * _minmax(df["peakToValley"])  # already negative-is-worse; minmax keeps that direction correct
        + 0.20 * _minmax(df["consistency"])
        + 0.10 * _minmax(df["annualizedReturn"])
        + 0.10 * _minmax(df["track_record"])
    )
    return df.sort_values("score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument("--account-nickname", default=None)
    args = parser.parse_args()

    api_key, user_key = _find_etoro_credentials(args.account_nickname)
    print("Fetching candidate investors from eToro rankings...")
    raw = fetch_candidates(api_key, user_key)
    print(f"  {len(raw)} unique popularInvestor candidates (riskScore <= {MAX_RISK_SCORE}) pulled")

    ranked = score_candidates(raw)
    print(
        f"  {len(ranked)} pass reliability filters (trades >= {MIN_TRADES}, "
        f"weeksSinceRegistration >= {MIN_WEEKS_SINCE_REGISTRATION})"
    )

    top = ranked.head(args.top)
    cols = [
        "username", "score", "annualizedReturn", "riskScore", "peakToValley",
        "winRatio", "profitableMonthsPct", "trades", "copiers", "weeksSinceRegistration",
    ]
    print(f"\nTop {args.top}:")
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(top[cols].to_string(index=False))

    print("\nUsernames only:", ", ".join(top["username"]))


if __name__ == "__main__":
    main()
