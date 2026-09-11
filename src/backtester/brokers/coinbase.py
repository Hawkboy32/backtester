"""Coinbase Advanced Trade broker implementation, backed by the official
coinbase-advanced-py SDK.

No verified sandbox/paper mode — every linked Coinbase account always
connects to the real live API (accounts.py forces is_paper=False for this
broker). Spot trading only. `submit_market_order`'s `ticker` accepts either
this app's usual Polygon-style ticker ("X:BTCUSD") or a real Coinbase
product_id ("BTC-USD") directly — see `_product_id()` for the translation.

Bracket orders (take_profit_price/stop_loss_price) aren't implemented —
Coinbase's trigger-bracket order shape doesn't map cleanly onto a simple
"market buy now, attach TP/SL" without further verification against a real
account, so passing those raises a clear error instead of guessing.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
from coinbase.rest import RESTClient

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position


def _get(obj, key):
    """Read `key` from a coinbase-advanced-py response field whether it's a
    plain dict, an attribute object, or an item-accessible object. The SDK is
    inconsistent — e.g. breakdown.portfolio_balances is a dict, but
    breakdown.spot_positions items are objects — so both styles must be handled."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    if hasattr(obj, key):
        return getattr(obj, key)
    try:
        return obj[key]
    except (TypeError, KeyError, IndexError):
        return None


def _money(field) -> float:
    """Extract a float from a Coinbase money field like {'value': '0', 'currency': 'GBP'}
    (dict or object), or a bare number. Returns 0.0 if absent/unparseable."""
    if field is None:
        return 0.0
    val = _get(field, "value") if not isinstance(field, (int, float)) else field
    try:
        return float(val) if val is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


class CoinbaseBroker(BrokerAccount):
    def __init__(self, nickname: str, api_key: str, api_secret: str, quote_currency: str = "USD"):
        self.nickname = nickname
        self.is_paper = False  # no verified sandbox for this broker
        self._client = RESTClient(api_key=api_key, api_secret=api_secret)
        # The currency this account actually HOLDS and EXECUTES in — default
        # USD (every ticker's own quote currency, matches its Polygon feed,
        # zero behavior change). An account funded in a different currency
        # (e.g. CBAPI holds GBP with no USD wallet at all — confirmed live
        # 2026-09-05, "account is not available" on every BTC-USD attempt)
        # can override this to trade its NATIVE product instead (BTC-GBP)
        # rather than needing a fiat conversion the exchange's own retail UI
        # doesn't offer for this account. See native_reference_price() below
        # for the other half this requires — pricing must follow currency
        # too, or sizing silently mis-fires.
        self.quote_currency = quote_currency

    def _product_id(self, ticker: str, quote_currency: str | None = None) -> str:
        """Coinbase's product_id format ("BTC-USD") differs from the Polygon-
        style ticker ("X:BTCUSD") used everywhere else in this app (universe
        CSVs, the Trade Execution ticker picker). Every pair in
        data/crypto_universe.csv is USD-quoted, so a strip+rejoin against
        the target quote currency is a reliable deterministic transform, not
        a guess. Already product-id-shaped input (contains a "-") passes
        through unchanged, so a hand-typed real product_id still works
        untouched.

        `quote_currency` defaults to `self.quote_currency` (the account's
        real EXECUTION/holding currency — GBP for CBAPI) for order placement
        and sizing. Callers that need the ticker's own USD-denominated
        market data instead (get_live_bars — see its docstring on why it
        must always be USD, not this account's currency) pass "USD"
        explicitly rather than relying on the default."""
        if "-" in ticker:
            return ticker
        t = ticker[2:] if ticker.startswith("X:") else ticker
        currency = quote_currency if quote_currency is not None else self.quote_currency
        return f"{t[:-3]}-{currency}" if t.endswith("USD") else ticker

    def native_reference_price(self, ticker: str, fallback_price: float) -> float:
        """Overrides the base no-op ONLY when this instance trades a
        currency other than the ticker's own (self.quote_currency != "USD").
        fallback_price is the strategy's USD signal price (e.g. Polygon
        X:BTCUSD) — correct for sizing an account that HOLDS USD, wrong for
        one that holds GBP: dividing GBP equity by a USD price would
        silently under-size every order by the GBP/USD rate with no error.

        Fetches Coinbase's own live price for the REAL executed product
        (e.g. BTC-GBP) directly, rather than converting via a separately-
        sourced FX rate — one live number from the venue that's about to
        fill the order, not a second data source that could disagree with
        it. Raises on failure rather than silently falling back to
        fallback_price: a sizing exception is caught by the caller and skips
        this cycle's trade (fails closed) — falling back here would silently
        reintroduce the exact currency-mismatch bug this method exists to
        prevent.
        """
        if self.quote_currency == "USD":
            return fallback_price
        product = self._client.get_product(product_id=self._product_id(ticker))
        return float(_get(product, "price"))

    def _default_portfolio_uuid(self) -> str:
        portfolios = self._client.get_portfolios().portfolios or []
        default = next((p for p in portfolios if p.type == "DEFAULT"), None)
        chosen = default or (portfolios[0] if portfolios else None)
        if not chosen:
            raise ValueError("No Coinbase portfolio found for this account")
        return chosen.uuid

    def get_account_snapshot(self) -> AccountSnapshot:
        portfolio_uuid = self._default_portfolio_uuid()
        breakdown = self._client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid).breakdown
        balances = _get(breakdown, "portfolio_balances")
        equity = _money(_get(balances, "total_balance"))
        cash = _money(_get(balances, "total_cash_equivalent_balance"))
        return AccountSnapshot(account_id=portfolio_uuid, equity=equity, cash=cash, buying_power=cash, is_paper=False)

    def get_positions(self) -> list[Position]:
        portfolio_uuid = self._default_portfolio_uuid()
        breakdown = self._client.get_portfolio_breakdown(portfolio_uuid=portfolio_uuid).breakdown
        positions = []
        for p in _get(breakdown, "spot_positions") or []:
            if _get(p, "is_cash"):
                continue  # skip fiat cash lines (e.g. GBP/USD) — not tradeable positions
            qty = float(_get(p, "total_balance_crypto") or 0.0)
            if qty == 0:
                continue  # dust-free: don't report zero-balance assets as positions
            market_value = float(_get(p, "total_balance_fiat") or 0.0)
            cost_basis = _money(_get(p, "cost_basis"))
            positions.append(
                Position(
                    ticker=_get(p, "asset"),
                    qty=qty,
                    side="long",
                    avg_entry_price=(cost_basis / qty) if cost_basis and qty else 0.0,
                    current_price=(market_value / qty) if qty else None,
                    market_value=market_value,
                    unrealized_pl=(market_value - cost_basis) if cost_basis else 0.0,
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        raise NotImplementedError(
            "Coinbase's Advanced Trade API doesn't expose a historical portfolio-equity time "
            "series through this SDK — only current balances (see get_account_snapshot)."
        )

    # (multiplier, timespan) -> Coinbase's own granularity constant, same
    # shape as alpaca.py's _TIMESPAN_TO_UNIT. Only the pairs this app's
    # get_live_bars callers actually use are mapped - an unmapped combo
    # raises clearly (see below) rather than guessing at a granularity
    # string, same "never guess" discipline as the rest of this file.
    _GRANULARITY = {
        (1, "minute"): "ONE_MINUTE",
        (5, "minute"): "FIVE_MINUTE",
        (15, "minute"): "FIFTEEN_MINUTE",
        (30, "minute"): "THIRTY_MINUTE",
        (1, "hour"): "ONE_HOUR",
        (1, "day"): "ONE_DAY",
    }

    # Coinbase's own hard cap (confirmed live 2026-08-20 via a real 400
    # INVALID_ARGUMENT response: "number of candles requested should be less
    # than 350") - 300 leaves a safety margin rather than riding the exact
    # documented limit.
    _MAX_CANDLES_PER_REQUEST = 300
    # Real ceiling on how many paginated requests one get_live_bars call will
    # make. auto_trader.py's LOOKBACK_DAYS=90 asks for far more than this can
    # ever cover at 1-minute granularity (90 days would need ~430 requests) -
    # deliberately NOT trying to serve that in full. The only strategy live
    # on crypto as of 2026-08-20 (VWAP Mean Reversion) needs 2 SESSIONS
    # (required_lookback(), vwap_mean_reversion.py) = 2 days at this
    # granularity - 15 requests * 300 candles = 4500 minutes = ~3.1 days,
    # comfortable margin over that real need without an unbounded loop.
    _MAX_PAGES = 15

    def get_live_bars(
        self, ticker: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = "minute",
    ) -> pd.DataFrame:
        """Live alternative to PolygonClient.get_aggregates() for the live
        trading loop specifically (see CLAUDE_NOTES.txt 2026-08-20: Polygon
        is being pulled out of the live path entirely, kept only for
        backtesting - Polygon's crypto data was found to run up to ~21h
        stale in practice, contradicting the earlier 2026-07-26 assumption
        that it had no gap). Shaped identically to Polygon/Alpaca's bars
        (columns: open/high/low/close/volume/vwap/transactions, indexed by
        UTC timestamp) so strategy code needs no changes.

        PAGINATES BACKWARD from `to_date`, in `_MAX_CANDLES_PER_REQUEST`-size
        chunks, up to `_MAX_PAGES` requests - NOT a general-purpose
        historical backfill (Polygon still owns that job for backtesting).
        If the requested (from_date, to_date) range needs more than
        _MAX_PAGES pages to fully cover, this returns whatever the most
        RECENT `_MAX_PAGES` pages contain rather than raising - a live
        decision should always prefer "some real recent data, missing the
        oldest requested days" over "no live data at all, silent fallback to
        a potentially stale Polygon read."

        Coinbase's candle response has no vwap or transactions field (unlike
        Polygon/Alpaca) - vwap is filled with close (a documented, honest
        approximation, not a guess at a real VWAP) and transactions with 0,
        since nothing in strategies/*.py's crypto-eligible strategies reads
        either field (VWAP Mean Reversion computes its own running VWAP from
        open/high/low/close/volume, never reads this column - confirmed by
        reading vwap_mean_reversion.py before relying on this).

        ALWAYS fetches the USD-quoted product, regardless of self.quote_currency
        - unlike native_reference_price/submit_market_order, which deliberately
        use the account's real execution currency (GBP for CBAPI). This
        method's whole contract is "shaped identically to Polygon's bars" for
        a Polygon-style ticker (X:BTCUSD always means USD, same as every other
        broker's get_live_bars) - every caller (strategy signal computation,
        the mobile app's chart) expects that, and silently handing back
        GBP-denominated candles under a "X:BTCUSD" label is a real, confirmed
        bug (found 2026-09-07: the mobile app's 5m/15m chart, which fetches
        bars fresh per-request and picked CBAPI, showed BTC at ~$58.7k -
        actually its GBP price - while the 1m view, using a cached signal
        snapshot computed via KRKAPI at the time, correctly showed ~$79.4k).
        """
        granularity = self._GRANULARITY.get((multiplier, timespan))
        if granularity is None:
            raise ValueError(f"Unsupported (multiplier, timespan) for Coinbase live bars: ({multiplier}, {timespan})")
        seconds_per_candle = {"ONE_MINUTE": 60, "FIVE_MINUTE": 300, "FIFTEEN_MINUTE": 900,
                               "THIRTY_MINUTE": 1800, "ONE_HOUR": 3600, "ONE_DAY": 86400}[granularity]
        chunk_span = seconds_per_candle * self._MAX_CANDLES_PER_REQUEST

        range_start = int(datetime.fromisoformat(from_date).replace(tzinfo=timezone.utc).timestamp())
        # +86400 makes to_date inclusive of its whole calendar day (matching
        # Polygon/Alpaca's own get_aggregates semantics) - but when to_date is
        # "today", midnight-today + 86400 = midnight-TOMORROW, which is in the
        # future relative to actual wall-clock time. Confirmed live 2026-08-21:
        # Coinbase's API validates start against its own server clock and
        # rejects it ("start must not be in the future") - capping at the real
        # current time fixes this without losing the inclusive-of-today
        # behavior for a genuinely past to_date.
        range_end = min(
            int(datetime.fromisoformat(to_date).replace(tzinfo=timezone.utc).timestamp()) + 86400,
            int(datetime.now(timezone.utc).timestamp()),
        )
        product_id = self._product_id(ticker, quote_currency="USD")
        columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]

        all_rows: list[dict] = []
        chunk_end = range_end
        for _ in range(self._MAX_PAGES):
            chunk_start = max(range_start, chunk_end - chunk_span)
            if chunk_start >= chunk_end:
                break
            response = self._client.get_candles(
                product_id=product_id, start=str(chunk_start), end=str(chunk_end), granularity=granularity,
            )
            for c in response.candles or []:
                close = float(c.close)
                all_rows.append({
                    "timestamp": pd.to_datetime(int(c.start), unit="s", utc=True),
                    "open": float(c.open), "high": float(c.high), "low": float(c.low),
                    "close": close, "volume": float(c.volume), "vwap": close, "transactions": 0,
                })
            if chunk_start <= range_start:
                break
            chunk_end = chunk_start

        if not all_rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(all_rows).drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        return df[columns]

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
                error="Bracket orders (take-profit/stop-loss) aren't implemented for Coinbase yet.",
            )
        product_id = self._product_id(ticker)
        try:
            client_order_id = str(uuid.uuid4())
            # retail_portfolio_id is required here, not optional decoration:
            # without it Coinbase can route the order against a portfolio
            # other than the one get_account_snapshot/get_positions read
            # from (confirmed live 2026-09-02 — orders failed with
            # INVALID_ARGUMENT "account is not available" while the same
            # default portfolio showed sufficient balance; matches
            # Coinbase's own documented multi-portfolio/legacy-key behavior).
            portfolio_uuid = self._default_portfolio_uuid()
            if side is OrderSide.BUY:
                response = self._client.market_order_buy(
                    client_order_id=client_order_id, product_id=product_id, base_size=str(qty),
                    retail_portfolio_id=portfolio_uuid,
                )
            else:
                response = self._client.market_order_sell(
                    client_order_id=client_order_id, product_id=product_id, base_size=str(qty),
                    retail_portfolio_id=portfolio_uuid,
                )
            if not response.success:
                error_msg = response.error_response.error if response.error_response else "unknown error"
                return OrderResult(account_nickname=self.nickname, success=False, error=str(error_msg))
            return OrderResult(account_nickname=self.nickname, success=True, broker_order_id=response.order_id)
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
