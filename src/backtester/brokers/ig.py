"""IG Group broker implementation, backed by the community `trading-ig` SDK
(REST wrapper around IG's documented dealing API).

Phase 3 scope (2026-07-27, agreed with the user before any code was written):
plain forex only via IG's CFD instruments — no spread-betting, no other CFD
markets (indices/commodities/shares-as-CFDs) built yet. Leverage is left
FULLY VARIABLE for backtesting/scanning (a pure historical-simulation
parameter, zero real risk at any level); LIVE arming requires its own
separate, explicit, hard-to-trigger confirmation — never inherited from
whatever leverage a backtest happened to use. The account-level risk limit
reuses the EXISTING account_risk.py drawdown breaker unchanged, per the
user's explicit choice to keep this simple now and revisit the threshold
once there's real data on how it behaves under margin — no separate
margin-aware breaker was built.

Session model: IG sessions expire after ~10 minutes (per IG's own docs, see
BROKER_EXPANSION_RESEARCH.md) — CONNECT-PER-OPERATION, same reasoning and
same pattern as ibkr.py, rather than holding a session across Streamlit
reruns where it would go stale.

TWO REAL GAPS, deliberately not guessed at (same discipline as every other
broker in this project — verify against a live account before assuming):
1. SYMBOL MAPPING — CLOSED 2026-08-03. IG identifies instruments by "epic"
   strings (e.g. "CS.D.EURUSD.CFD.IP" is the commonly-documented EUR/USD CFD
   epic — never independently confirmed here; _resolve_epic() below resolves
   the real epic via search rather than assuming it). `submit_market_order` now calls
   `_resolve_epic()`, which resolves a plain pair name (or this app's usual
   Polygon-style "C:EURUSD" ticker) via IGService's own `search_markets()`
   — the real SDK method for this, previously unused here — rather than
   assuming any hardcoded epic pattern. LIVE-VERIFIED (see gap 2): a real
   "C:EURUSD" order was accepted by the linked IG demo account. An already-
   epic-shaped ticker (matches the same heuristic accounts.infer_asset_class
   uses) skips resolution entirely, so a hand-typed real epic still works.
2. LIVE-VERIFIED 2026-08-03: `_resolve_epic()`'s search_markets()-based
   resolution was exercised for real against the linked IG demo account —
   a manual "C:EURUSD" order (the exact ticker that originally failed with
   validation.pattern.invalid.request.epic before this method existed) was
   accepted. The `instrumentType`/`epic` column names it filters on are
   confirmed correct for at least this pair, not just plausible.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone

import pandas as pd
from trading_ig import IGService
from trading_ig.rest import IGException

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position
from backtester.brokers.ibkr import _forex_clock_heuristic

_DEMO_BASE_URL = "https://demo-api.ig.com/gateway/deal"
_LIVE_BASE_URL = "https://api.ig.com/gateway/deal"

# INVESTIGATED 2026-07-30 — a per-minute-quota theory was tried first and
# RULED OUT: IG Labs' own current FAQ (labs.ig.com/faq.html) publishes 30
# non-trading requests/minute per account (60/minute per app), and a manually-
# spaced test (one call every 20s, well under that) still failed. Then the
# IG API companion tool (labs.ig.com/companion) logged in with these EXACT
# credentials via a normal browser and got a clean HTTP 200 — proving the
# account, password, and API key are all fine, and ruling out an account-
# level block too. What's left: this app's own automated client most likely
# got itself flagged by IG's security layer after a burst of rapid, repeated
# login attempts during testing (mine and the dashboard's) — a common anti-
# abuse response, separate from the published quota, that a normal browser
# session doesn't trigger. NOT independently confirmed with IG (would need
# their WebAPI support) — treat as the best-supported explanation, not a
# proven mechanism. _last_ig_call_at / _IG_THROTTLE_LOCK keep a conservative
# minimum gap between this process's own IG calls regardless — cheap
# insurance, but retrying repeatedly on failure is exactly the behavior that
# may have caused this, so _session() below no longer retries by default.
_TRANSIENT_AUTH_ERROR = "service.security.authentication.failure-invalid-client-security-token"
_MIN_SECONDS_BETWEEN_IG_CALLS = 20.0
_IG_THROTTLE_LOCK = threading.Lock()

# Bars requested per get_live_bars() call. Every point billed against the
# 10,000/week allowance (see get_live_bars' own docstring), so this is
# deliberately the SMALLEST window that still covers what the only current
# caller needs: Opening Spike Fade reads from the session open through its
# reversal window, ~25 minutes, and 45 leaves headroom for a late start or a
# missing bar without paying for a whole session's history every cycle.
_IG_BARS_NUMPOINTS = 45
# IG returns tz-naive timestamps in the ACCOUNT's local time - UK for this
# account. Verified live 2026-08-25 against a known UTC clock; see
# get_live_bars for the arithmetic. Named here so the assumption is
# reviewable in one place rather than buried in the conversion.
_IG_BARS_TZ = "Europe/London"
_last_ig_call_at = 0.0


def _throttle_ig_call() -> None:
    global _last_ig_call_at
    with _IG_THROTTLE_LOCK:
        wait = _last_ig_call_at + _MIN_SECONDS_BETWEEN_IG_CALLS - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_ig_call_at = time.monotonic()


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


# Currency-CFD epics follow "CS.D.<PAIR>.<TYPE>.IP" (e.g. "CS.D.EURUSD.CFD.IP"
# / "CS.D.EURUSD.MINI.IP") — the middle segment IS the plain 6-letter pair
# name, so it can be parsed straight back into this app's usual Polygon-style
# ticker ("C:EURUSD") without needing _resolve_epic's in-memory cache (which
# is empty on every fresh process start, so a cache-only reverse lookup would
# still return raw epics for any position opened before THIS instance
# existed — a real gap for a long-running bot that gets restarted).
_EPIC_PAIR_RE = re.compile(r"^CS\.D\.([A-Z]{6})\.")


def _ticker_from_epic(epic: str) -> str:
    """Best-effort reverse of _resolve_epic: a real currency-CFD epic ->
    this app's "C:<PAIR>" ticker form. Falls back to the raw epic unchanged
    for anything that doesn't match the expected forex-CFD shape (e.g. a
    position IG shows that this app didn't open/resolve itself) — never
    raises, since get_positions() must not fail just because one row's
    epic looks unfamiliar."""
    match = _EPIC_PAIR_RE.match(epic)
    return f"C:{match.group(1)}" if match else epic


def _round_to_ig_size_step(qty: float, min_size: float) -> float:
    """Round DOWN to the nearest size IG will actually accept.

    Originally assumed "any multiple of dealingRules.minDealSize" — WRONG,
    confirmed live 2026-08-04: on an account/instrument reporting
    minDealSize=1.0, a request for size=4 was rejected with SIZE_INCREMENT,
    while size=1 and size=10 both passed the size check (size=10 got only
    as far as a separate INSUFFICIENT_FUNDS rejection, proving 10 itself
    was accepted as a valid size). That pattern — 1 and 10 valid, 4 not —
    matches a "1-2-5" preferred-number lot-size sequence (1, 2, 5, 10, 20,
    50, 100, ...), a common convention on CFD platforms that ISN'T exposed
    anywhere in the dealingRules API response — it was inferred from
    limited live testing, not documented, so treat it as a working
    hypothesis to revisit if a future order still comes back SIZE_INCREMENT,
    not settled fact. Never rounds up (never over-commits beyond what was
    requested/sized) — returns 0.0 if qty is below the smallest step.
    """
    if min_size <= 0 or qty < min_size:
        return 0.0
    best = 0.0
    step = min_size
    while step <= qty:
        for multiple in (1, 2, 5):
            candidate = step * multiple
            if candidate > qty:
                return best
            best = candidate
        step *= 10
    return best


# REVISED 2026-07-31: the throttle above wasn't enough — a clean login
# followed by ANOTHER fresh login shortly after (even 20s+ later, even with
# no retries) reliably failed, while the SAME already-authenticated session
# kept working. That matches IG's own docs (see labs.ig.com/rest-trading-api-
# guide.html): v1/v2 CST/X-SECURITY-TOKEN tokens are valid for 6-72 hours,
# extended on every use — NOT the ~10 minutes this file originally assumed
# (that figure applies to v3's OAuth tokens, which this broker doesn't use).
# So the real fix is to stop treating "new session per call" as free just
# because IBKR's connect-per-operation model does — for IG specifically, hold
# ONE session per IGBroker instance and reuse it, only re-authenticating when
# there isn't one yet or a call proves it's gone stale. Known remaining gap:
# build_broker_accounts() constructs a fresh IGBroker (and so a fresh session)
# on every call — this helps any caller that holds one IGBroker instance
# across multiple operations (this module's own methods, ad-hoc scripts), but
# a caller that rebuilds broker instances per Streamlit rerun / per poll cycle
# still pays for a new login each time. Caching BrokerAccount instances across
# reruns is a separate, bigger change, not done here.
_SESSION_MAX_AGE_SECONDS = 5.0 * 3600  # conservative vs IG's documented 6h floor


class IGBroker(BrokerAccount):
    def __init__(
        self,
        nickname: str,
        username: str,
        password: str,
        api_key: str,
        is_paper: bool = True,
        ig_account_id: str = "",
        use_encryption: bool = False,
    ):
        self.nickname = nickname
        self.is_paper = is_paper
        self._username = username
        self._password = password
        self._api_key = api_key
        self._ig_account_id = ig_account_id or None
        # IG's own SDK: password encryption is "Required for some regions" —
        # off by default (matches the SDK's own default), on when a plaintext
        # login is rejected. See create_session's use below.
        self._use_encryption = use_encryption
        self._svc: IGService | None = None
        self._svc_created_at: float = 0.0
        # Resolved-epic cache, keyed by whatever ticker was passed in. Epics
        # are stable for a given instrument, so this persists for this
        # instance's lifetime — same reasoning as caching the session itself.
        self._epic_cache: dict[str, str] = {}
        # Per-epic minimum-deal-size/increment cache (see _resolve_min_deal_size)
        # — also stable per instrument, same caching reasoning as the epic cache.
        self._min_deal_size_cache: dict[str, float] = {}

    def _new_session(self) -> IGService:
        """Always creates a brand-new, freshly-authenticated IGService — does
        NOT retry on failure (see the module-level note on why repeated
        automated login attempts are themselves suspect), and is throttled
        against this process's own last IG call regardless of caller."""
        _throttle_ig_call()
        svc = IGService(
            username=self._username,
            password=self._password,
            api_key=self._api_key,
            acc_type="demo" if self.is_paper else "live",
            acc_number=self._ig_account_id,
        )
        svc.create_session(encryption=self._use_encryption)
        return svc

    def _session(self) -> IGService:
        """Returns this instance's cached, already-authenticated session,
        creating one only if there isn't one yet or it's old enough that IG
        might have expired it. Callers must NOT log out of this — logging out
        would invalidate it for every other method sharing this instance.
        Use _new_session() directly only when a fresh, throwaway session is
        actually wanted (there's no legitimate case for that in this file)."""
        if self._svc is None or (time.monotonic() - self._svc_created_at) > _SESSION_MAX_AGE_SECONDS:
            self._svc = self._new_session()
            self._svc_created_at = time.monotonic()
        return self._svc

    def close(self) -> None:
        """Explicitly ends the cached session, if any. Not required for
        correctness (IG's own tokens simply age out), but lets a caller that's
        done with this broker instance free the session deliberately."""
        if self._svc is not None:
            try:
                self._svc.logout()
            except IGException:
                pass
            finally:
                self._svc = None

    def _resolve_epic(self, ticker: str) -> str:
        """Resolve a plain pair name (or this app's usual Polygon-style
        "C:EURUSD" ticker) to a real IG epic via IGService.search_markets()
        — the SDK's own instrument-search endpoint — rather than assuming
        any hardcoded epic pattern (see the module docstring's gap 1: the
        commonly-cited "CS.D.<PAIR>.CFD.IP" shape has never been confirmed
        against a real account, so it isn't hardcoded here as if it were).

        Already epic-shaped input (same heuristic accounts.infer_asset_class
        uses to detect a resolved epic) passes through unchanged — a hand-
        typed real epic still works, and no API call is wasted on it. Results
        are cached per-instance since an epic doesn't change.
        """
        if ticker.startswith("CS.D.") or ticker.startswith("IX.D.") or ".CFD." in ticker or ".MINI." in ticker:
            return ticker
        if ticker in self._epic_cache:
            return self._epic_cache[ticker]

        # Index tickers (Polygon-style "I:NDX") don't share forex's clean
        # "strip the prefix, search the raw pair" mapping - "NDX" alone
        # doesn't reliably find IG's "US Tech 100" product via search_markets.
        # An explicit table, confirmed live 2026-08-23/24 against a real
        # search_markets("US Tech 100") call (epic IX.D.NASDAQ.IFD.IP,
        # instrumentType INDICES) rather than guessed - only I:NDX is mapped
        # since that's the only index this app trades so far; add entries
        # here as more come up, same spirit as kraken.py's
        # _KRAKEN_SYMBOL_OVERRIDES for the same kind of naming mismatch.
        index_map = {"I:NDX": "US Tech 100", "I:SPX": "US 500"}
        pair = index_map.get(ticker, ticker[2:] if ticker.startswith("C:") else ticker)
        svc = self._session()
        results = svc.search_markets(pair)
        if results is None or results.empty:
            raise IGException(f"IG search_markets found no instrument matching {pair!r} (from ticker {ticker!r}).")

        # Prefer a plain currency or index CFD match (IG's instrumentType for
        # forex pairs / indices respectively) — fall back to the first result
        # if that filter finds nothing, rather than failing outright on an
        # unexpected type value.
        preferred_types = {"CURRENCIES", "INDICES"}
        typed_rows = results[results["instrumentType"].isin(preferred_types)] if "instrumentType" in results else results
        candidates = typed_rows if not typed_rows.empty else results
        if ticker in index_map:
            # Prefer the plain "Cash" product over futures/weekend variants
            # that also match the same name search (confirmed live
            # 2026-08-24: a "US Tech 100" search returns Cash, futures
            # (FWS2/FWM2 epics), AND a "Weekend US Tech 100" row all at
            # once - "Cash" in the instrument name is the reliable filter,
            # the row ORDER alone isn't a safe assumption to rely on).
            cash_rows = candidates[candidates["instrumentName"].str.contains("Cash", na=False)]
            if not cash_rows.empty:
                candidates = cash_rows
        row = candidates.iloc[0]
        epic = str(row["epic"])
        self._epic_cache[ticker] = epic
        return epic

    def _resolve_min_deal_size(self, epic: str) -> float:
        """The smallest size IG will accept for this instrument, fetched via
        IGService.fetch_market_by_epic()'s real dealingRules.minDealSize.value
        field (confirmed against the trading-ig SDK's own response shape,
        not guessed) and cached per-instance alongside _epic_cache — stable
        per instrument, not worth re-fetching every order. Falls back to 1.0
        if the field is missing/unparseable. NOTE: this is the FLOOR, not
        the step — see _round_to_ig_size_step's docstring for why a clean
        multiple of this value can still be rejected.
        """
        if epic in self._min_deal_size_cache:
            return self._min_deal_size_cache[epic]
        details = self._session().fetch_market_by_epic(epic)
        try:
            size = float(details["dealingRules"]["minDealSize"]["value"])
            if size <= 0:
                size = 1.0
        except (KeyError, TypeError, ValueError):
            size = 1.0
        self._min_deal_size_cache[epic] = size
        return size

    def get_account_snapshot(self) -> AccountSnapshot:
        svc = self._session()
        accounts = svc.fetch_accounts()
        row = accounts.iloc[0] if self._ig_account_id is None else (
            accounts[accounts["accountId"] == self._ig_account_id].iloc[0]
        )
        return AccountSnapshot(
            account_id=str(row["accountId"]),
            equity=_num(row.get("balance")),
            cash=_num(row.get("available")),
            buying_power=_num(row.get("available")),
            is_paper=self.is_paper,
        )

    def get_positions(self) -> list[Position]:
        svc = self._session()
        df = svc.fetch_open_positions()
        positions: list[Position] = []
        for _, row in df.iterrows():
            qty = _num(row.get("size"))
            if qty == 0:
                continue
            entry = _num(row.get("level"))
            bid, offer = _num(row.get("bid")), _num(row.get("offer"))
            current = (bid + offer) / 2 if (bid or offer) else None
            positions.append(
                Position(
                    ticker=_ticker_from_epic(str(row.get("epic"))),
                    qty=qty,
                    side="long" if str(row.get("direction", "")).upper() == "BUY" else "short",
                    avg_entry_price=entry,
                    current_price=current,
                    market_value=qty * (current or entry),
                    unrealized_pl=qty * ((current or entry) - entry),
                )
            )
        return positions

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        # fetch_transaction_history_by_type_and_period exists but returns
        # individual TRANSACTIONS, not a reconstructed equity time series —
        # mapping one to the other correctly needs verifying against real
        # account activity first (same reasoning Coinbase/Kraken/IBKR give
        # for the same NotImplementedError). The balances chart already
        # skips brokers that raise this rather than fabricating a curve.
        raise NotImplementedError(
            "IG's REST API doesn't expose a ready-made portfolio-equity time series through "
            "this SDK — only current balance (see get_account_snapshot) and raw transaction "
            "history, which hasn't been verified as reconstructible into an equity curve yet."
        )

    def get_live_bars(
        self, ticker: str, from_date: str, to_date: str, multiplier: int = 1, timespan: str = "minute",
    ) -> pd.DataFrame:
        """Live bars from IG's historical-price endpoint, shaped identically
        to Polygon/Alpaca/Coinbase's (open/high/low/close/volume/vwap/
        transactions, indexed by UTC timestamp) so strategy code needs no
        changes. Added 2026-08-25 specifically to get INDEX tickers (I:NDX)
        off Polygon, which Alpaca rejects outright ("invalid symbol: I:NDX")
        and which therefore silently fell back to Polygon - the one live
        target still doing so after the 2026-08-20 Polygon-is-backtesting-
        only migration.

        *** READ THIS BEFORE CALLING IT ANYWHERE NEW ***
        IG's historical-price endpoint has a hard 10,000-data-points-per-WEEK
        allowance (confirmed live 2026-08-25: totalAllowance 10000, a rolling
        604799s expiry). That is far too scarce for an unconditional 120s
        polling loop - ~195 cycles a session at even 60 points each is
        ~11,700 points in ONE DAY - which is exactly why forex uses OANDA
        rather than IG for data (see auto_trader._live_data_account_for's own
        docstring, where an early test pull exhausted the allowance outright).
        Callers MUST therefore bound how often they call this; see
        auto_trader._index_bars_window_open() for the time-window gate that
        makes the NDX case fit (~1,800 points/week).

        Deliberately NO caching/retry here: the allowance is consumed by the
        REQUEST, so a silent retry doubles the cost of a bad day.
        """
        if timespan != "minute":
            raise ValueError(f"IG live bars only support minute bars here, got {timespan!r}")
        epic = self._resolve_epic(ticker)
        # numpoints rather than a date range: the endpoint bills per point
        # returned, so asking for exactly what's needed is the difference
        # between fitting the weekly allowance and blowing it in a morning.
        resolution = f"{multiplier}min"
        numpoints = _IG_BARS_NUMPOINTS
        resp = self._session().fetch_historical_prices_by_epic_and_num_points(
            epic, resolution, numpoints
        )
        frame = resp["prices"]
        if frame is None or frame.empty:
            return pd.DataFrame()

        # IG quotes an index as bid/ask; the mid is the honest single price
        # (Polygon's I:NDX is the index level itself, which sits between the
        # two), and using bid alone would bias every level a spread low.
        def _mid(field: str) -> pd.Series:
            return (frame[("bid", field)] + frame[("ask", field)]) / 2.0

        out = pd.DataFrame(
            {
                "open": _mid("Open"),
                "high": _mid("High"),
                "low": _mid("Low"),
                "close": _mid("Close"),
            }
        )
        # ("last", "Volume") is IG's traded-contract count for the bar. Index
        # epics do report it; it is NOT a share volume and nothing here
        # treats it as one - it only feeds the liquidity/slippage model's
        # relative sizing, same as every other source's volume column.
        try:
            out["volume"] = frame[("last", "Volume")].astype(float)
        except KeyError:
            out["volume"] = 0.0
        out["volume"] = out["volume"].fillna(0.0)
        # No vwap/transactions from IG - filled to match the shared column
        # contract, exactly as data.py does for Polygon's own index tickers.
        out["vwap"] = 0.0
        out["transactions"] = 0.0

        # IG returns tz-NAIVE timestamps in the account's local time, which is
        # UK for this account - verified live 2026-08-25 by arithmetic against
        # a known UTC clock (IG bar 08:37 vs 07:37Z = +1, i.e. Europe/London
        # in BST). Localising through the zone rather than a fixed +1 keeps it
        # correct across the GMT/BST switch.
        idx = pd.to_datetime(out.index)
        if idx.tz is None:
            idx = idx.tz_localize(_IG_BARS_TZ, ambiguous="infer", nonexistent="shift_forward")
        out.index = idx.tz_convert("UTC")
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out.dropna(subset=["open", "high", "low", "close"])

    def get_market_clock(self) -> dict | None:
        # No IG account exists yet to verify a real per-epic market-status
        # lookup (fetch_market_by_epic's marketStatus field is the documented
        # path, but needs a resolved epic and a live account to confirm the
        # exact values it returns). Reuses IBKR's already-tested forex weekly
        # heuristic (Sunday 17:00 ET - Friday 17:00 ET) since IG's forex CFDs
        # track real spot forex hours closely enough for this to be a
        # reasonable approximation — same spirit as ibkr.py's own heuristic
        # fallback, just promoted to the ONLY path here until IG's live
        # market-status endpoint is verified and epics are resolved.
        return _forex_clock_heuristic()

    def get_open_order_tickers(self) -> set[str]:
        # IG's OTC market orders (create_open_position) fill SYNCHRONOUSLY —
        # create_open_position blocks until fetch_deal_by_deal_reference
        # returns a confirmation, so there is no "submitted but not yet
        # filled" state to dedupe against for a plain market order the way
        # IBKR/Alpaca need. fetch_working_orders() covers LIMIT/STOP orders,
        # which this broker doesn't submit (submit_market_order is market-
        # only, matching every other broker's interface here) — empty set is
        # the correct default, not a punt.
        return set()

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        """`ticker` accepts either this app's usual Polygon-style ticker
        ("C:EURUSD"), a plain pair name, or a real IG epic directly —
        _resolve_epic() translates as needed. Opens a new position (long via
        BUY, short via SELL — see account_position_modes) if none exists on
        the resolved epic; closes the existing one otherwise, in whichever
        direction actually closes it (opposite of the held direction, not
        tied to the `side` argument — see the close branch below). Made
        explicit here since IG's API has separate open/close endpoints
        instead of one order call the broker nets automatically.
        """
        svc = self._session()
        try:
            ticker = self._resolve_epic(ticker)

            min_size = self._resolve_min_deal_size(ticker)
            rounded_qty = _round_to_ig_size_step(qty, min_size)
            if rounded_qty <= 0:
                return OrderResult(
                    account_nickname=self.nickname, success=False,
                    error=(
                        f"Requested size {qty} is below {ticker}'s minimum tradeable size "
                        f"({min_size}) — raise the sizing value."
                    ),
                )
            qty = rounded_qty

            # Fetch the RAW positions frame here, not get_positions()'s
            # translated Position list — closing needs IG's own dealId,
            # which the generic Position dataclass (shared across every
            # broker in this project) has no field for.
            raw_positions = svc.fetch_open_positions()
            existing_row = None
            if not raw_positions.empty:
                matches = raw_positions[raw_positions["epic"] == ticker]
                existing_row = matches.iloc[0] if not matches.empty else None

            try:
                if existing_row is None:
                    # A BUY opens a long, a SELL opens a short — the caller
                    # (auto_trader.py) only ever sends SELL here for an
                    # account explicitly configured short_only/long_short
                    # (account_position_modes); every other account stays
                    # long-only via that gating, not this method rejecting
                    # anything itself. IG's own API treats both identically
                    # (create_open_position with direction=BUY or SELL), so
                    # there's nothing IG-specific blocking a short here.
                    direction = "BUY" if side is OrderSide.BUY else "SELL"
                    confirm = svc.create_open_position(
                        currency_code="USD", direction=direction, epic=ticker, expiry="-",
                        force_open=True, guaranteed_stop=False, level=None,
                        limit_distance=None,
                        limit_level=(take_profit_price if take_profit_price is not None else None),
                        order_type="MARKET", quote_id=None, size=qty,
                        stop_distance=None,
                        stop_level=(stop_loss_price if stop_loss_price is not None else None),
                        trailing_stop=False, trailing_stop_increment=None,
                    )
                else:
                    # Closing direction is the OPPOSITE of the held position
                    # (documented IG convention, now verified live 2026-07-31).
                    # REAL BUG FOUND + FIXED same day: IG's close endpoint
                    # rejects a request identifying the position by BOTH
                    # dealId and epic/expiry at once (HTTP 400
                    # validation.mutual-exclusive-value.request) — a position
                    # close must be identified ONE way, and dealId is the
                    # more precise of the two, so epic/expiry are left unset
                    # here rather than passed alongside it.
                    close_direction = "SELL" if str(existing_row.get("direction", "")).upper() == "BUY" else "BUY"
                    confirm = svc.close_open_position(
                        deal_id=str(existing_row.get("dealId")), direction=close_direction,
                        epic=None, expiry=None, level=None, order_type="MARKET",
                        quote_id=None, size=qty,
                    )
            except IGException as e:
                return OrderResult(account_nickname=self.nickname, success=False, error=str(e))

            status = str(confirm.get("dealStatus", "")).upper()
            if status != "ACCEPTED":
                reason = confirm.get("reason") or f"IG dealStatus={status or 'unknown'}"
                return OrderResult(
                    account_nickname=self.nickname, success=False,
                    broker_order_id=confirm.get("dealId"), error=str(reason),
                )
            return OrderResult(
                account_nickname=self.nickname, success=True,
                broker_order_id=confirm.get("dealId"),
                filled_qty=_num(confirm.get("size"), qty),
                filled_avg_price=_num(confirm.get("level")) or None,
            )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))
