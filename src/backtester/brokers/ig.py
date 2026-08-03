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

import threading
import time
from datetime import datetime, timezone

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
        if ticker.startswith("CS.D.") or ".CFD." in ticker or ".MINI." in ticker:
            return ticker
        if ticker in self._epic_cache:
            return self._epic_cache[ticker]

        pair = ticker[2:] if ticker.startswith("C:") else ticker
        svc = self._session()
        results = svc.search_markets(pair)
        if results is None or results.empty:
            raise IGException(f"IG search_markets found no instrument matching {pair!r} (from ticker {ticker!r}).")

        # Prefer a plain currency CFD match (IG's instrumentType for forex
        # pairs) — fall back to the first result if that filter finds
        # nothing, rather than failing outright on an unexpected type value.
        currency_rows = results[results["instrumentType"] == "CURRENCIES"] if "instrumentType" in results else results
        row = (currency_rows if not currency_rows.empty else results).iloc[0]
        epic = str(row["epic"])
        self._epic_cache[ticker] = epic
        return epic

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
                    ticker=str(row.get("epic")),
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
        _resolve_epic() translates as needed. Opens a new position if none
        exists on the resolved epic; closes the existing one otherwise —
        mirrors how the rest of this app's auto-trading logic already
        decides BUY-to-open vs SELL-to-close (long-only everywhere, no
        shorting), just made explicit here since IG's API has separate
        open/close endpoints instead of one order call the broker nets
        automatically.
        """
        svc = self._session()
        try:
            ticker = self._resolve_epic(ticker)
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
                    if side is not OrderSide.BUY:
                        return OrderResult(
                            account_nickname=self.nickname, success=False,
                            error=f"No open position on {ticker} to sell — long-only, nothing to close.",
                        )
                    confirm = svc.create_open_position(
                        currency_code="USD", direction="BUY", epic=ticker, expiry="-",
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
