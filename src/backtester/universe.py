"""Ticker universe loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SP500_CSV_PATH = DATA_DIR / "sp500_constituents.csv"
NASDAQ100_CSV_PATH = DATA_DIR / "nasdaq100_constituents.csv"


def _load_csv(path: Path, refresh_script: str, max_tickers: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run {refresh_script} first.")
    df = pd.read_csv(path)
    if max_tickers is not None:
        df = df.head(max_tickers)
    return df


def load_sp500(max_tickers: int | None = None) -> pd.DataFrame:
    """Return the cached S&P 500 constituent list (ticker, name, sector, sub_industry).

    This is a periodic snapshot (see data/sp500_constituents.meta.txt for the
    fetch date), not a live feed — refresh it with scripts/refresh_sp500_list.py
    if it's gone stale.
    """
    return _load_csv(SP500_CSV_PATH, "scripts/refresh_sp500_list.py", max_tickers)


def load_nasdaq100(max_tickers: int | None = None) -> pd.DataFrame:
    """Return the cached Nasdaq-100 constituent list (a.k.a. "USA Tech 100" /
    "US Tech 100" in some brokers' naming). Same periodic-snapshot caveat as
    load_sp500 — refresh with scripts/refresh_nasdaq100_list.py if stale.
    """
    return _load_csv(NASDAQ100_CSV_PATH, "scripts/refresh_nasdaq100_list.py", max_tickers)


def load_sp500_tickers(max_tickers: int | None = None) -> list[str]:
    return load_sp500(max_tickers)["ticker"].tolist()


UNIVERSE_REGISTRY = {
    "S&P 500": load_sp500,
    "Nasdaq-100 (US Tech 100)": load_nasdaq100,
}


def load_universe(name: str, max_tickers: int | None = None) -> pd.DataFrame:
    if name not in UNIVERSE_REGISTRY:
        raise ValueError(f"Unknown universe '{name}'. Available: {list(UNIVERSE_REGISTRY)}")
    return UNIVERSE_REGISTRY[name](max_tickers)
