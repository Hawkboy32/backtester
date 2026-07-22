"""Polygon.io market data client for fetching historical aggregate bars.

Includes a conservative rate limiter (tier unknown by default — assume the
free tier's ~5 requests/minute unless told otherwise) and an on-disk cache
so multi-ticker scans are resumable and don't re-fetch data they already have.
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.massive.com"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cache" / "bars"


class PolygonError(RuntimeError):
    pass


class RateLimiter:
    """Thread-safe sliding-window rate limiter."""

    def __init__(self, requests_per_minute: int = 5):
        self.requests_per_minute = requests_per_minute
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._calls and now - self._calls[0] > 60:
                    self._calls.popleft()
                if len(self._calls) < self.requests_per_minute:
                    self._calls.append(now)
                    return
                sleep_for = 60 - (now - self._calls[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))


class PolygonClient:
    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        requests_per_minute: int = 5,
        cache_dir: Path | str | None = DEFAULT_CACHE_DIR,
        use_cache: bool = True,
    ):
        self.api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not self.api_key:
            raise PolygonError(
                "No Polygon API key found. Set POLYGON_API_KEY in your environment or .env file."
            )
        self.session = session or requests.Session()
        self.rate_limiter = RateLimiter(requests_per_minute)
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.use_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, params: dict | None = None, max_attempts: int = 3) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            self.rate_limiter.wait()
            try:
                resp = self.session.get(url, params=params, headers=headers, timeout=30)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                # Transient network failures (e.g. the server closing a stale
                # keep-alive connection mid-scan) — retry with backoff rather
                # than aborting a long multi-ticker scan over one hiccup.
                last_error = e
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                continue
            if resp.status_code == 429:
                raise PolygonError("Rate limited by Polygon API (HTTP 429). Slow down requests.")
            if resp.status_code >= 500:
                # Transient server-side failure — retry like a network error.
                last_error = PolygonError(f"Polygon server error HTTP {resp.status_code}")
                if attempt < max_attempts:
                    time.sleep(2 * attempt)
                continue
            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                # Non-retryable client error (4xx) — surface as PolygonError so
                # callers' per-ticker error handling catches it instead of
                # aborting a whole scan.
                raise PolygonError(f"Polygon API error: {e}") from e
            return resp.json()
        raise PolygonError(
            f"Network error talking to Polygon after {max_attempts} attempts: {last_error}"
        ) from last_error

    def _cache_path(self, cache_key: str) -> Path:
        digest = hashlib.sha256(cache_key.encode()).hexdigest()[:24]
        return self.cache_dir / f"{digest}.pkl"

    def get_aggregates(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        multiplier: int = 1,
        timespan: str = "minute",
        adjusted: bool = True,
        sort: str = "asc",
        limit: int = 50000,
        pause_between_pages: float = 0.0,
    ) -> pd.DataFrame:
        """Fetch OHLCV aggregate bars for a ticker over a date range, following pagination.

        Dates are 'YYYY-MM-DD'. Returns a DataFrame indexed by UTC timestamp with
        columns: open, high, low, close, volume, vwap, transactions.

        Results are cached on disk (keyed by all params) unless use_cache=False.
        """
        cache_key = f"{ticker}|{multiplier}|{timespan}|{from_date}|{to_date}|{adjusted}|{sort}"
        if self.use_cache and self.cache_dir:
            cache_path = self._cache_path(cache_key)
            if cache_path.exists():
                return pd.read_pickle(cache_path)

        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {"adjusted": str(adjusted).lower(), "sort": sort, "limit": limit}

        rows: list[dict] = []
        next_url: str | None = url
        next_params: dict | None = params
        while next_url:
            payload = self._get(next_url, next_params)
            status = payload.get("status")
            if status not in ("OK", "DELAYED"):
                raise PolygonError(f"Unexpected response status: {payload}")
            rows.extend(payload.get("results", []) or [])
            next_url = payload.get("next_url")
            next_params = None  # next_url already carries its own query params
            if next_url and pause_between_pages:
                time.sleep(pause_between_pages)

        if not rows:
            df = pd.DataFrame(columns=["open", "high", "low", "close", "volume", "vwap", "transactions"])
        else:
            df = pd.DataFrame(rows).rename(
                columns={
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                    "vw": "vwap",
                    "n": "transactions",
                    "t": "timestamp_ms",
                }
            )
            df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[["open", "high", "low", "close", "volume", "vwap", "transactions"]]

        if self.use_cache and self.cache_dir:
            df.to_pickle(self._cache_path(cache_key))

        return df
