"""OANDA v20 API client for live forex candle data.

Exists because neither of this project's other two data sources works for
forex's LIVE decision loop: this account's Polygon plan has no same-day
intraday data at all (confirmed 2026-08-05, see CLAUDE_NOTES.txt), and IG's
own historical-price endpoint has a real weekly data-point allowance that one
test pull exhausted in a single call — far too scarce for a 120s polling
loop. OANDA's practice account offers free, generously-rate-limited REST
access to real forex candles with no such quota (confirmed via OANDA's own
docs before building this).

Used HERE ONLY for DATA — this is deliberately NOT a BrokerAccount subclass
and has no order/position capability. IG remains the only broker that ever
places a forex order; OANDA is never a target account (see
auto_trader.py's _pick_live_data_source). For a pair as liquid as EUR/USD,
prices across major forex venues track each other extremely closely — far
tighter than the same-instrument gap you'd see comparing two different stock
exchanges — so trading on IG's own fills while deciding off OANDA's quote is
a reasonable, low-risk split, not a data/execution mismatch.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

PRACTICE_BASE_URL = "https://api-fxpractice.oanda.com"
LIVE_BASE_URL = "https://api-fxtrade.oanda.com"

# OANDA caps every single request at 5000 candles (confirmed via OANDA's own
# docs) regardless of whether you ask by count or by from/to range — a 90-day
# lookback at 1-minute granularity is ~90,000+ candles, so this MUST paginate
# or it silently truncates to whatever the first page happens to cover.
MAX_CANDLES_PER_REQUEST = 5000
MAX_PAGES = 200  # safety cap against an unexpected response shape looping forever

_GRANULARITY_MAP = {
    ("minute", 1): "M1",
    ("minute", 5): "M5",
    ("minute", 15): "M15",
    ("minute", 30): "M30",
    ("hour", 1): "H1",
    ("day", 1): "D",
}


class OandaError(RuntimeError):
    pass


def _to_oanda_instrument(ticker: str) -> str:
    """"C:EURUSD" -> "EUR_USD" — this project's Polygon-style forex ticker to
    OANDA's underscore-separated instrument name. Assumes a standard 6-letter
    pair after stripping the "C:" prefix (matches every forex ticker this
    project already uses, e.g. C:EURUSD, C:GBPUSD)."""
    pair = ticker[2:] if ticker.startswith("C:") else ticker
    if len(pair) != 6:
        raise OandaError(f"Can't convert {ticker!r} to an OANDA instrument (expected a 6-letter pair)")
    return f"{pair[:3]}_{pair[3:]}"


class OandaDataClient:
    nickname = "OANDA"

    def __init__(
        self,
        api_key: str | None = None,
        practice: bool = True,
        session: requests.Session | None = None,
    ):
        self.api_key = api_key or os.environ.get("OANDA_API_KEY")
        if not self.api_key:
            raise OandaError(
                "No OANDA API key found. Set OANDA_API_KEY in your environment or .env file."
            )
        self.base_url = PRACTICE_BASE_URL if practice else LIVE_BASE_URL
        self.session = session or requests.Session()

    def _get_page(self, instrument: str, granularity: str, from_iso: str, to_iso: str) -> list[dict]:
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        params = {
            "price": "M",  # midpoint — one clean OHLC series, not separate bid/ask
            "granularity": granularity,
            "from": from_iso,
            "to": to_iso,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        resp = self.session.get(url, params=params, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise OandaError(f"OANDA API error {resp.status_code}: {resp.text[:300]}")
        return resp.json().get("candles", [])

    def get_live_bars(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        multiplier: int = 1,
        timespan: str = "minute",
    ) -> pd.DataFrame:
        """Same shape as PolygonClient.get_aggregates() / AlpacaBroker.get_live_bars():
        columns open/high/low/close/volume/vwap/transactions, UTC-indexed —
        so strategy code needs zero changes to consume this instead of either.

        Dates are 'YYYY-MM-DD'. Paginates internally past OANDA's 5000-candle
        per-request cap, advancing the from-cursor past the last COMPLETE
        candle each page (see MAX_CANDLES_PER_REQUEST above).
        """
        granularity = _GRANULARITY_MAP.get((timespan, multiplier))
        if granularity is None:
            raise OandaError(f"Unsupported timespan/multiplier for OANDA: {timespan}/{multiplier}")

        instrument = _to_oanda_instrument(ticker)
        # OANDA rejects a `to` timestamp that's in the future outright (unlike
        # Polygon, which silently just returns whatever's actually available)
        # — found live 2026-08-06: requesting through 23:59:59 of TODAY errors
        # with "Invalid value specified for 'to'. Time is in the future" any
        # time before the last second of the UTC day, which is effectively
        # always. Cap at the real current moment instead.
        end_of_day = pd.Timestamp(f"{to_date}T23:59:59Z")
        now = pd.Timestamp.now(tz="UTC")
        to_iso = min(end_of_day, now).strftime("%Y-%m-%dT%H:%M:%SZ")
        cursor = f"{from_date}T00:00:00Z"

        rows: list[dict] = []
        seen_times: set[str] = set()
        for _ in range(MAX_PAGES):
            candles = self._get_page(instrument, granularity, cursor, to_iso)
            if not candles:
                break

            new_last_time = None
            for c in candles:
                if not c.get("complete", True):
                    continue  # skip the still-forming current candle — same "closed bar only" contract as Polygon/Alpaca
                t = c["time"]
                if t in seen_times:
                    continue  # OANDA's next page overlaps its own last candle
                seen_times.add(t)
                new_last_time = t
                mid = c["mid"]
                o, h, l, close = float(mid["o"]), float(mid["h"]), float(mid["l"]), float(mid["c"])
                volume = float(c.get("volume", 0))
                rows.append({
                    "timestamp": pd.Timestamp(t, tz="UTC") if pd.Timestamp(t).tzinfo is None else pd.Timestamp(t).tz_convert("UTC"),
                    "open": o, "high": h, "low": l, "close": close,
                    "volume": volume,
                    "vwap": (o + h + l + close) / 4,  # OANDA has no vwap field — simple 4-point approximation
                    "transactions": int(volume),  # OANDA's "volume" is tick count, the closest analog to Polygon's transactions
                })

            if len(candles) < MAX_CANDLES_PER_REQUEST or new_last_time is None:
                break  # last page (fewer than the cap came back, or nothing new — done)
            cursor = new_last_time

        columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return df[columns]
