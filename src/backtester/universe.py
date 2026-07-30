"""Ticker universe loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SP500_CSV_PATH = DATA_DIR / "sp500_constituents.csv"
NASDAQ100_CSV_PATH = DATA_DIR / "nasdaq100_constituents.csv"
CRYPTO_CSV_PATH = DATA_DIR / "crypto_universe.csv"
FOREX_CSV_PATH = DATA_DIR / "forex_universe.csv"


def sample_universe(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n rows spread evenly across the full (alphabetically-sorted) frame,
    rather than the first n. head(n) on a 503-row, alphabetically-sorted CSV
    means every S&P 500 scan ever run only covered tickers starting with "A"
    (confirmed 2026-07-26 — every recorded scan's max_tickers<=50 never once
    touched a ticker past "AVY") — the mean-reversion edge those scans found
    is real WITHIN that slice but has never been tested against ~90% of the
    index. Deterministic (not random) so a given max_tickers is reproducible
    scan to scan, same as before — just representative of the whole list
    instead of one alphabetic corner of it. Public: callers that already
    hold a full-universe DataFrame (e.g. to size a UI slider) and need to
    re-slice it by a chosen count without re-reading the CSV should use this
    directly, rather than DataFrame.head().
    """
    if n >= len(df):
        return df
    idx = np.unique(np.round(np.linspace(0, len(df) - 1, num=n)).astype(int))
    # linspace rounding can collapse two targets onto the same integer index
    # when n is a large fraction of len(df) — top up from the nearest unused
    # rows so the caller always gets exactly n, never fewer.
    if len(idx) < n:
        remaining = np.setdiff1d(np.arange(len(df)), idx)
        idx = np.sort(np.concatenate([idx, remaining[: n - len(idx)]]))
    return df.iloc[idx]


def _load_csv(path: Path, refresh_script: str, max_tickers: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run {refresh_script} first.")
    df = pd.read_csv(path)
    if max_tickers is not None:
        df = sample_universe(df, max_tickers)
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


def load_crypto(max_tickers: int | None = None) -> pd.DataFrame:
    """Hand-curated top-15 USD pairs (ticker, name, sector, sub_industry) — see
    data/crypto_universe.meta.txt for why this can't be scraped like the equity
    universes, and for the Polygon-ticker-vs-broker-symbol gap that still needs
    solving before this feeds real execution (scanning/backtesting only for now).
    """
    return _load_csv(CRYPTO_CSV_PATH, "(hand-edit data/crypto_universe.csv)", max_tickers)


def load_forex(max_tickers: int | None = None) -> pd.DataFrame:
    """Hand-curated 7 major USD currency pairs (ticker, name, sector,
    sub_industry) — see data/forex_universe.meta.txt for why this can't be
    scraped like the equity universes, why it needs its OWN Sharpe
    annualization calendar distinct from crypto (metrics.MARKET_CALENDARS
    "forex" — 24h session but closed weekends, unlike crypto's real 24/7),
    and for the IBKR execution gap that still needs solving before this feeds
    real orders (scanning/backtesting only for now).
    """
    return _load_csv(FOREX_CSV_PATH, "(hand-edit data/forex_universe.csv)", max_tickers)


UNIVERSE_REGISTRY = {
    "S&P 500": load_sp500,
    "Nasdaq-100 (US Tech 100)": load_nasdaq100,
    "Crypto (top 15 USD pairs)": load_crypto,
    "Forex (7 major USD pairs)": load_forex,
}


def load_universe(name: str, max_tickers: int | None = None) -> pd.DataFrame:
    if name not in UNIVERSE_REGISTRY:
        raise ValueError(f"Unknown universe '{name}'. Available: {list(UNIVERSE_REGISTRY)}")
    return UNIVERSE_REGISTRY[name](max_tickers)


@lru_cache(maxsize=1)
def _sector_map() -> dict[str, str]:
    """ticker -> sector across every universe CSV we have. Built once per
    process. Note the two CSVs use different sector taxonomies (S&P's GICS vs
    the Nasdaq page's own labels, e.g. "Information Technology" vs
    "Technology"), so a ticker in both keeps the FIRST one loaded — good enough
    for "don't put the whole roster in one sector", which is all this feeds.
    """
    mapping: dict[str, str] = {}
    for loader in UNIVERSE_REGISTRY.values():
        try:
            df = loader(None)
        except FileNotFoundError:
            continue
        if "sector" not in df.columns:
            continue
        for ticker, sector in zip(df["ticker"], df["sector"]):
            if isinstance(sector, str) and sector and ticker not in mapping:
                mapping[str(ticker)] = sector
    return mapping


def sector_for_ticker(ticker: str) -> str | None:
    """The ticker's sector, or None if we don't know it (e.g. a hand-typed
    ticker outside both universes). Callers must treat None as "unconstrained"
    rather than lumping unknowns together into a fake shared sector.
    """
    return _sector_map().get(ticker.upper())
