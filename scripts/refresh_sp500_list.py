"""Refresh the local S&P 500 constituent list.

Pulls the constituents table from Wikipedia's "List of S&P 500 companies"
page using pandas.read_html (a structural HTML-table parse, not an LLM
summary) and writes it to data/sp500_constituents.csv.

Index membership changes periodically (a handful of times a year) — re-run
this script occasionally rather than treating the CSV as permanently
accurate.

Usage:
    python scripts/refresh_sp500_list.py
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sp500_constituents.csv"


def main() -> None:
    resp = requests.get(
        WIKI_URL,
        headers={"User-Agent": "backtester-project/0.1 (personal research script)"},
        timeout=30,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    constituents = tables[0]

    df = constituents.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
        }
    )[["ticker", "name", "sector", "sub_industry"]]

    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False).str.strip()
    df = df.sort_values("ticker").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    fetched_at = datetime.now(timezone.utc).isoformat()
    meta_path = OUTPUT_PATH.with_suffix(".meta.txt")
    meta_path.write_text(f"fetched_at_utc={fetched_at}\nsource={WIKI_URL}\ncount={len(df)}\n")

    print(f"Wrote {len(df)} constituents to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
