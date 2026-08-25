"""Kraken broker implementation, backed by the low-level krakenex SDK
(krakenex only wraps signing/nonce — response shapes match Kraken's REST
API docs directly, not a typed client).

No verified sandbox/paper mode — every linked Kraken account always
connects to the real live API (accounts.py forces is_paper=False for this
broker). Spot trading only. `submit_market_order`'s `ticker` accepts either
this app's usual Polygon-style ticker ("X:BTCUSD") or a real Kraken altname
pair ("XBTUSD") directly — see `_pair()` for the translation.

Position market values aren't computed: Kraken's asset-code system (XXBT,
XETH, ZUSD, ...) doesn't map onto trading pairs in a small, reliably-correct
way without a full pair-lookup table, so get_positions() reports quantity
only rather than guess at a price. Account-level equity in
get_account_snapshot is exact — Kraken computes it server-side (TradeBalance
"eb" field), no guessing involved there.

Bracket orders (take_profit_price/stop_loss_price) aren't implemented.
"""

from __future__ import annotations

from datetime import datetime, timezone

import krakenex
import pandas as pd

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position

FIAT_ASSETS = {"ZUSD", "ZEUR", "ZGBP", "ZCAD", "ZJPY", "ZCHF", "ZAUD"}

# Kraken's altname pair codes for every symbol in data/crypto_universe.csv,
# confirmed 2026-08-19 against a live query of Kraken's own public
# AssetPairs endpoint (not guessed) — most of the 15 map cleanly by
# strip+rejoin (Polygon's "X:ADAUSD" -> Kraken altname "ADAUSD", etc.), but
# two use Kraken's legacy asset codes and would silently resolve to the
# wrong (or a nonexistent) pair under a naive strip+rejoin: Bitcoin is "XBT"
# not "BTC" (Kraken predates BTC's ticker convention), and Dogecoin is "XDG"
# not "DOGE" (DOGEUSD exists on Kraken but is a DIFFERENT, unrelated pair —
# confirmed via the same live query, base="DOG" not Dogecoin). Every other
# symbol's altname was verified to equal the plain strip+rejoin form
# (ETHUSD, SOLUSD, XRPUSD, ADAUSD, AVAXUSD, LINKUSD, DOTUSD, LTCUSD,
# BCHUSD, UNIUSD, ATOMUSD, XLMUSD, ETCUSD), so no full 15-entry table is
# needed — just the two real exceptions.
_KRAKEN_SYMBOL_OVERRIDES = {"BTC": "XBT", "DOGE": "XDG"}


class KrakenError(RuntimeError):
    pass


class KrakenBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, api_secret: str):
        self.nickname = nickname
        self.is_paper = False  # no verified sandbox for this broker
        self._client = krakenex.API(key=api_key, secret=api_secret)

    def _pair(self, ticker: str) -> str:
        """Coinbase-style translation (see coinbase.py's _product_id) from
        this app's Polygon ticker format to Kraken's altname pair code.
        Already Kraken-shaped input (no "X:" prefix, or containing no
        recognizable Polygon pattern) passes through unchanged, so a
        hand-typed real Kraken pair still works untouched."""
        if not ticker.startswith("X:"):
            return ticker
        t = ticker[2:]
        if not t.endswith("USD"):
            return ticker
        symbol = t[:-3]
        return f"{_KRAKEN_SYMBOL_OVERRIDES.get(symbol, symbol)}USD"

    def _private(self, method: str, data: dict | None = None) -> dict:
        response = self._client.query_private(method, data)
        if response.get("error"):
            raise KrakenError("; ".join(response["error"]))
        return response.get("result", {})

    def get_account_snapshot(self) -> AccountSnapshot:
        balance = self._private("Balance")
        trade_balance = self._private("TradeBalance")
        cash = float(balance.get("ZUSD", 0.0))
        equity = float(trade_balance.get("eb", cash))
        return AccountSnapshot(account_id=self.nickname, equity=equity, cash=cash, buying_power=cash, is_paper=False)

    def get_positions(self) -> list[Position]:
        balance = self._private("Balance")
        positions = []
        for asset, qty_str in balance.items():
            qty = float(qty_str)
            if asset in FIAT_ASSETS or qty == 0:
                continue
            positions.append(
                Position(
                    ticker=asset,
                    qty=qty,
                    side="long",
                    avg_entry_price=0.0,
                    current_price=None,
                    market_value=0.0,
                    unrealized_pl=0.0,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        raise NotImplementedError(
            "Kraken's API doesn't expose a historical portfolio-equity time series through "
            "this client — only current balances (see get_account_snapshot)."
        )

    # (multiplier, timespan) -> Kraken's OHLC interval, in minutes - same
    # shape/reasoning as coinbase.py's _GRANULARITY (added alongside it,
    # 2026-08-20). Kraken's own valid intervals per its API docs.
    _INTERVAL_MINUTES = {
        (1, "minute"): 1,
        (5, "minute"): 5,
        (15, "minute"): 15,
        (30, "minute"): 30,
        (1, "hour"): 60,
        (1, "day"): 1440,
    }

    def get_live_bars(
        self, ticker: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = "minute",
    ) -> pd.DataFrame:
        """Same contract as coinbase.py's get_live_bars (see that docstring
        for the fuller "why this exists" context, incl. Polygon being pulled
        out of the live path entirely, 2026-08-20) - added alongside it once
        KRKAPI went live 2026-08-20.

        Kraken's public OHLC endpoint (query_public, no auth needed, though
        this reuses the account's already-authenticated client for
        simplicity) returns each row as [time, open, high, low, close,
        vwap, volume, count] - confirmed live 2026-08-19 while building the
        _pair() ticker-mapping fix, not guessed. Unlike Coinbase's candle
        endpoint, Kraken's DOES include real vwap and trade-count fields, so
        neither needs approximating here.

        REAL, CONFIRMED LIMITATION (tested live 2026-08-20, not assumed from
        docs): `since` does NOT page backward the way Coinbase's start/end
        does - a request with since=3 days ago still only returned the most
        recent ~721 candles ending at "now" (~12h of 1-minute data), not 3
        days starting from since. There is no working way to pull an older
        window from this endpoint. Coinbase's get_live_bars (proper
        start/end pagination, up to ~3+ days) is therefore the one actually
        capable of covering VWAP Mean Reversion's 2-session lookback need on
        its own - _pick_live_data_source's ordering (first eligible account
        wins) naturally prefers Coinbase for that reason, since CBAPI is
        listed before KRKAPI in every extra_targets entry touching crypto.
        This method still has real value on its own: it keeps KRKAPI's own
        signal path off of stale Polygon data for whatever it CAN cover
        (~12h), it just isn't the primary bars source for a 2-day-lookback
        strategy. `to_date` is applied as a client-side filter after
        fetching, since the API gives no server-side way to bound it.
        """
        interval = self._INTERVAL_MINUTES.get((multiplier, timespan))
        if interval is None:
            raise ValueError(f"Unsupported (multiplier, timespan) for Kraken live bars: ({multiplier}, {timespan})")

        since = int(datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc).timestamp())
        until = int(datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc).timestamp()) + 86400
        pair = self._pair(ticker)
        columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]

        response = self._client.query_public("OHLC", {"pair": pair, "interval": interval, "since": since})
        if response.get("error"):
            raise KrakenError("; ".join(response["error"]))
        result = response.get("result", {})
        data_key = next((k for k in result if k != "last"), None)
        rows_raw = result.get(data_key, []) if data_key else []
        if not rows_raw:
            return pd.DataFrame(columns=columns)

        rows = []
        for time_s, o, h, low, c, vwap, vol, count in rows_raw:
            if time_s > until:
                continue
            rows.append({
                "timestamp": pd.to_datetime(int(time_s), unit="s", utc=True),
                "open": float(o), "high": float(h), "low": float(low),
                "close": float(c), "volume": float(vol), "vwap": float(vwap), "transactions": int(count),
            })
        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows).set_index("timestamp").sort_index()[columns]

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        if take_profit_price is not None or stop_loss_price is not None:
            return OrderResult(
                account_nickname=self.nickname,
                success=False,
                error="Bracket orders (take-profit/stop-loss) aren't implemented for Kraken yet.",
            )
        try:
            result = self._private(
                "AddOrder",
                {
                    "pair": self._pair(ticker),
                    "type": "buy" if side is OrderSide.BUY else "sell",
                    "ordertype": "market",
                    "volume": str(qty),
                },
            )
            txids = result.get("txid", [])
            return OrderResult(
                account_nickname=self.nickname,
                success=True,
                broker_order_id=txids[0] if txids else None,
            )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
