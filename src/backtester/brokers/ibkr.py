"""Interactive Brokers implementation, backed by the ib_async library (the
maintained successor to ib_insync) talking to a locally-running IB Gateway or
Trader Workstation.

IBKR is the odd one out among this app's brokers: there is NO API key/secret.
The bot connects to a gateway process the user runs and logs into themselves;
authentication lives in that gateway. So an IBKR account is defined by
connection config — host, port, clientId, and (optionally) the IBKR account
code — NOT by stored secrets. accounts.py keeps these as plain metadata, never
in the OS keyring.

Connection model: CONNECT-PER-OPERATION. ib_async is asyncio-based, while this
app is synchronous and rerun-heavy (Streamlit) and also runs inside the separate
auto_trader.py process. Rather than hold a long-lived connection across reruns
(which goes stale and complicates clientId reuse), every method opens a short
connection, does its work, and disconnects in a finally. Each op ensures an
event loop exists in the current thread first (Streamlit's ScriptRunner thread
and the auto_trader thread don't have one by default), and uses a jittered
clientId so a not-yet-released previous connection doesn't cause "client id
already in use".

Scope (task #30): US-equity execution, same instruments as Alpaca, reusing the
existing Polygon equities data path. Forex/CFD instruments were out of scope
at first — spot forex (real, unleveraged) added 2026-07-27 (Phase 2) via the
asset_class="forex" constructor param below, gated explicitly per-account
rather than inferred from a ticker string (a bare 6-letter pair like "EURUSD"
could in principle collide with an unusual equity ticker; the broker instance
itself knows which asset class it represents, so nothing has to guess).
"""

from __future__ import annotations

import asyncio
import random
from contextlib import contextmanager
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ib_async import IB, Forex, LimitOrder, MarketOrder, StopOrder, Stock

from backtester.brokers.base import AccountSnapshot, BrokerAccount, EquityPoint, OrderResult, OrderSide, Position

# Reference symbol used only to read the US-equity trading calendar from IBKR
# (its contract details carry liquidHours/timeZoneId). Any liquid US listing works.
_CLOCK_SYMBOL = "SPY"
# Reference forex pair for the same purpose when asset_class="forex" — IBKR
# reports the same liquidHours/timeZoneId mechanism for a CASH contract.
_FOREX_CLOCK_PAIR = "EURUSD"
_DEFAULT_TIMEOUT = 8.0
ASSET_CLASSES = ("equity", "forex")


def _ensure_event_loop() -> None:
    """ib_async needs an asyncio event loop in the current thread. Streamlit's
    script thread and the auto_trader worker thread don't have one — create it
    on demand so IB() can be constructed off the main thread."""
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


class IBKRBroker(BrokerAccount):
    def __init__(
        self,
        nickname: str,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 1,
        ibkr_account: str = "",
        is_paper: bool = True,
        asset_class: str = "equity",
    ):
        if asset_class not in ASSET_CLASSES:
            raise ValueError(f"Unknown asset_class {asset_class!r}. Available: {ASSET_CLASSES}")
        self.nickname = nickname
        self.is_paper = is_paper
        self._host = host
        self._port = int(port)
        self._client_id_base = int(client_id)
        self._ibkr_account = ibkr_account or ""
        self._asset_class = asset_class

    @property
    def supports_fractional_shares(self) -> bool:
        """IBKR's STK order path rejects fractional-sized equity orders
        outright — confirmed live 2026-08-03: "Fractional-sized order cannot
        be placed via API. Please use desktop version to place this order."
        (Alpaca supports fractional shares; IBKR doesn't via this API.) Forex
        (CASH) orders aren't measured in discrete shares, so this only
        applies to an equity-flavored account."""
        return self._asset_class != "equity"

    def _contract(self, ticker: str):
        """A Stock or Forex contract depending on this account's asset_class —
        NOT inferred from `ticker`'s shape, set once at construction."""
        if self._asset_class == "forex":
            return Forex(ticker)
        return Stock(ticker, "SMART", "USD")

    @contextmanager
    def _connect(self, timeout: float = _DEFAULT_TIMEOUT, readonly: bool = False):
        """Open a short-lived IB connection, yield it, and always disconnect.
        clientId is jittered off the configured base so a previous connection
        that hasn't fully released doesn't collide."""
        _ensure_event_loop()
        ib = IB()
        client_id = self._client_id_base + random.randint(0, 999)
        try:
            ib.connect(
                host=self._host,
                port=self._port,
                clientId=client_id,
                timeout=timeout,
                readonly=readonly,
                account=self._ibkr_account,
            )
            yield ib
        finally:
            try:
                ib.disconnect()
            except Exception:  # noqa: BLE001 — never let a disconnect hiccup mask the real result
                pass

    def _values_by_tag(self, ib: IB) -> dict[str, str]:
        # accountValues() is populated by the ACCOUNT_UPDATES fetch on connect.
        rows = ib.accountValues(self._ibkr_account)
        # Prefer the base-currency rows; IBKR reports some tags per-currency.
        out: dict[str, str] = {}
        for v in rows:
            if v.currency in ("", "BASE", "USD") or v.tag not in out:
                out[v.tag] = v.value
        return out

    def get_account_snapshot(self) -> AccountSnapshot:
        with self._connect() as ib:
            tags = self._values_by_tag(ib)
            account = self._ibkr_account or (ib.managedAccounts()[0] if ib.managedAccounts() else "")

            def _num(tag: str) -> float:
                try:
                    return float(tags.get(tag, "0") or "0")
                except (TypeError, ValueError):
                    return 0.0

            # BuyingPower is the standard tag; fall back to AvailableFunds if absent.
            buying_power = _num("BuyingPower") or _num("AvailableFunds")
            return AccountSnapshot(
                account_id=account,
                equity=_num("NetLiquidation"),
                cash=_num("TotalCashValue"),
                buying_power=buying_power,
                is_paper=self.is_paper,
            )

    def get_positions(self) -> list[Position]:
        with self._connect() as ib:
            items = ib.portfolio(self._ibkr_account)
            positions: list[Position] = []
            for it in items:
                if not it.position:
                    continue  # skip flat/closed rows
                # A CASH (forex) contract's .symbol is only the base currency
                # (e.g. "EUR") — reconstruct the full pair like the rest of
                # this app names it ("EURUSD"), same reasoning as below.
                ticker = (
                    it.contract.symbol + it.contract.currency
                    if it.contract.secType == "CASH"
                    else it.contract.symbol
                )
                positions.append(
                    Position(
                        ticker=ticker,
                        qty=float(it.position),
                        side="long" if it.position > 0 else "short",
                        avg_entry_price=float(it.averageCost),
                        current_price=float(it.marketPrice) if it.marketPrice is not None else None,
                        market_value=float(it.marketValue) if it.marketValue is not None else 0.0,
                        unrealized_pl=float(it.unrealizedPNL) if it.unrealizedPNL is not None else 0.0,
                    )
                )
            return positions

    def get_market_clock(self) -> dict | None:
        """Real trading calendar for this account's asset class, sourced from
        IBKR's own contract details (handles holidays/half-days). This MUST be
        accurate: the auto_trader's market-hours guard (the overnight-overtrade
        fix) relies on it — returning None would make IBKR look 24/7-open like
        a crypto venue, which is wrong for BOTH equities (real close) and forex
        (closed weekends despite its ~24h weekday session). Falls back to a
        plain weekday-hours heuristic only if the IBKR hours string can't be
        parsed."""
        try:
            with self._connect(readonly=True) as ib:
                probe = (
                    Forex(_FOREX_CLOCK_PAIR) if self._asset_class == "forex"
                    else Stock(_CLOCK_SYMBOL, "SMART", "USD")
                )
                details = ib.reqContractDetails(probe)
                if details:
                    cd = details[0]
                    parsed = _parse_ib_hours(cd.liquidHours, cd.timeZoneId)
                    if parsed is not None:
                        return parsed
        except Exception:  # noqa: BLE001 — fall through to the heuristic below
            pass
        return _forex_clock_heuristic() if self._asset_class == "forex" else _us_equity_clock_heuristic()

    def get_open_order_tickers(self) -> set[str]:
        with self._connect(readonly=True) as ib:
            trades = ib.reqAllOpenOrders()
            active = {"PendingSubmit", "PreSubmitted", "Submitted", "ApiPending", "PendingCancel"}
            out = set()
            for t in trades:
                if not (t.orderStatus and t.orderStatus.status in active and t.contract and t.contract.symbol):
                    continue
                out.add(
                    t.contract.symbol + t.contract.currency
                    if t.contract.secType == "CASH"
                    else t.contract.symbol
                )
            return out

    def get_equity_history(self, period: str = "1M", timeframe: str = "1D") -> list[EquityPoint]:
        # The TWS API has no simple portfolio-equity time series like Alpaca's
        # get_portfolio_history. The balances chart already skips brokers that
        # raise this (same as Coinbase/Kraken), rather than fabricating a curve.
        raise NotImplementedError("IBKR (TWS API) does not expose a portfolio-equity history series.")

    def submit_market_order(
        self,
        ticker: str,
        side: OrderSide,
        qty: float,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> OrderResult:
        try:
            with self._connect() as ib:
                contract = self._contract(ticker)
                ib.qualifyContracts(contract)
                action = "BUY" if side is OrderSide.BUY else "SELL"

                # The REAL rejection reason (e.g. 202 "No market data ... for
                # market order") arrives on the error event channel, not in the
                # trade log (which may only carry cosmetic notices like the 10349
                # TIF preset message). Collect errors for this order as they land.
                order_errors: list[tuple[int, int, str]] = []
                ib.errorEvent += lambda reqId, code, msg, *a: order_errors.append((reqId, code, msg))

                if take_profit_price is None and stop_loss_price is None:
                    parent_trade = ib.placeOrder(contract, MarketOrder(action, qty))
                else:
                    parent_trade = self._place_market_bracket(
                        ib, contract, action, qty, take_profit_price, stop_loss_price
                    )

                # IBKR delivers rejections ASYNCHRONOUSLY on its error channel —
                # placeOrder() not raising proves nothing. (Found in live paper
                # testing: error 202 "No market data on major exchange for market
                # order" cancelled the order ~1s after a clean submit.) Watch the
                # order briefly and treat a cancelled/inactive outcome as failure,
                # surfacing the real reason from the trade log. An order still
                # merely queued (PreSubmitted etc.) after the wait is a success —
                # it lives on IBKR's servers after we disconnect.
                dead_states = {"Cancelled", "ApiCancelled", "Inactive"}
                for _ in range(5):
                    ib.sleep(1)
                    if parent_trade.orderStatus.status in dead_states or parent_trade.isDone():
                        break
                status = parent_trade.orderStatus
                if status.status in dead_states:
                    # The explanatory error event (e.g. 202) can land SECONDS after
                    # the Cancelled status (observed ~3-4s in live paper testing) —
                    # poll up to 4 more seconds, stopping as soon as it arrives.
                    oid = parent_trade.order.orderId
                    for _ in range(4):
                        if any(rid == oid and code != 10349 for rid, code, _m in order_errors):
                            break
                        ib.sleep(1)
                    # Prefer the error-channel reason for THIS order (skipping the
                    # cosmetic 10349 TIF notice); fall back to the trade log.
                    reason = next(
                        (msg for rid, code, msg in reversed(order_errors) if rid == oid and code != 10349),
                        None,
                    ) or next(
                        (e.message for e in reversed(parent_trade.log) if e.message),
                        f"order ended {status.status} without a stated reason",
                    )
                    return OrderResult(
                        account_nickname=self.nickname,
                        success=False,
                        broker_order_id=str(parent_trade.order.orderId),
                        error=f"IBKR {status.status}: {reason}",
                    )
                filled = float(status.filled) if status and status.filled else None
                avg = float(status.avgFillPrice) if status and status.avgFillPrice else None
                return OrderResult(
                    account_nickname=self.nickname,
                    success=True,
                    broker_order_id=str(parent_trade.order.orderId),
                    filled_qty=filled,
                    filled_avg_price=avg,
                )
        except Exception as e:  # noqa: BLE001
            return OrderResult(account_nickname=self.nickname, success=False, error=str(e))

    def _place_market_bracket(self, ib, contract, action, qty, take_profit_price, stop_loss_price):
        """Market-parent bracket. ib_async's bracketOrder() helper forces a LIMIT
        parent, so we build the trio by hand: a market entry (transmit=False) plus
        the opposite-side take-profit / stop-loss children linked by parentId, with
        transmit=True on the last child to fire the whole bracket atomically."""
        reverse = "SELL" if action == "BUY" else "BUY"
        parent = MarketOrder(action, qty)
        parent.orderId = ib.client.getReqId()
        parent.transmit = False

        children = []
        if take_profit_price is not None:
            tp = LimitOrder(reverse, qty, take_profit_price)
            tp.orderId = ib.client.getReqId()
            tp.parentId = parent.orderId
            tp.transmit = False
            children.append(tp)
        if stop_loss_price is not None:
            sl = StopOrder(reverse, qty, stop_loss_price)
            sl.orderId = ib.client.getReqId()
            sl.parentId = parent.orderId
            sl.transmit = False
            children.append(sl)
        # The final order transmitted must carry transmit=True to release the bracket.
        children[-1].transmit = True

        parent_trade = ib.placeOrder(contract, parent)
        for child in children:
            ib.placeOrder(contract, child)
        return parent_trade


def check_gateway_reachable(host: str, port: int, timeout: float = 3.0) -> bool:
    """Lightweight reachability probe for the Accounts-tab status chip: can we
    open an API session to the gateway at all? A fast, read-only connect that we
    immediately drop. False on any failure (gateway down, API port not enabled,
    wrong port, etc.)."""
    _ensure_event_loop()
    ib = IB()
    try:
        ib.connect(host=host, port=int(port), clientId=random.randint(1, 999), timeout=timeout, readonly=True)
        return ib.isConnected()
    except Exception:  # noqa: BLE001
        return False
    finally:
        try:
            ib.disconnect()
        except Exception:  # noqa: BLE001
            pass


def _parse_ib_hours(liquid_hours: str, tz_id: str) -> dict | None:
    """Parse IBKR's liquidHours string (e.g.
    '20260725:0930-20260725:1600;20260726:CLOSED') in its timeZoneId into a
    {'is_open': bool, 'next_open': datetime} dict. Returns None if it can't be
    parsed, so the caller can fall back to a heuristic."""
    if not liquid_hours or not tz_id:
        return None
    try:
        tz = ZoneInfo(tz_id)
    except Exception:  # noqa: BLE001 — unknown tz id
        return None

    now = datetime.now(tz)
    open_intervals: list[tuple[datetime, datetime]] = []
    saw_any_segment = False  # a CLOSED day is still a valid, informative segment
    for segment in liquid_hours.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if segment.endswith("CLOSED"):
            saw_any_segment = True  # IBKR is explicitly telling us this day is closed (e.g. a holiday)
            continue
        try:
            start_s, end_s = segment.split("-")
            start = _parse_ib_stamp(start_s, tz)
            # End may be 'YYYYMMDD:HHMM' or just 'HHMM' (same day as start).
            end = _parse_ib_stamp(end_s, tz, default_date=start)
            open_intervals.append((start, end))
            saw_any_segment = True
        except Exception:  # noqa: BLE001 — skip any unparseable segment
            continue

    # Only bail to the heuristic if we couldn't read ANY segment. If we saw
    # segments but none are open right now (weekend / holiday), that's a
    # definitive "closed" — crucial so a holiday isn't mistaken for open.
    if not saw_any_segment:
        return None

    open_intervals.sort()
    is_open = any(start <= now <= end for start, end in open_intervals)
    future_opens = [start for start, _ in open_intervals if start > now]
    next_open = min(future_opens) if future_opens else None
    return {"is_open": is_open, "next_open": next_open}


def _parse_ib_stamp(stamp: str, tz: ZoneInfo, default_date: datetime | None = None) -> datetime:
    stamp = stamp.strip()
    if ":" in stamp:
        date_part, time_part = stamp.split(":")
        dt = datetime.strptime(date_part + time_part, "%Y%m%d%H%M")
    else:
        # Bare HHMM — reuse the start segment's date.
        base = default_date or datetime.now(tz)
        dt = base.replace(hour=int(stamp[:2]), minute=int(stamp[2:]), second=0, microsecond=0)
        return dt if dt.tzinfo else dt.replace(tzinfo=tz)
    return dt.replace(tzinfo=tz)


def _us_equity_clock_heuristic() -> dict:
    """Fallback US-equity clock: NYSE regular session 09:30-16:00 America/New_York,
    Mon-Fri. Does NOT know holidays (would report a holiday as open) — used only
    when IBKR's own hours string can't be read. Approximate, flagged as such."""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    session_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    session_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    is_weekday = now.weekday() < 5
    is_open = is_weekday and session_open <= now <= session_close

    if is_open:
        next_open = session_open
    else:
        candidate = session_open if (is_weekday and now < session_open) else session_open + timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        next_open = candidate
    return {"is_open": is_open, "next_open": next_open}


def _forex_clock_heuristic() -> dict:
    """Fallback forex clock: one continuous ~24h/weekday session from Sunday
    17:00 to Friday 17:00 America/New_York (the standard forex week — NOT the
    equity pattern of daily closes) with no daily close in between. Does NOT
    know forex-specific holidays (thin Christmas/New Year liquidity etc.) —
    used only when IBKR's own hours string can't be read. Approximate, flagged
    as such, same spirit as _us_equity_clock_heuristic."""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    week_close = now.replace(hour=17, minute=0, second=0, microsecond=0)  # Friday 17:00 ends the week

    # weekday(): Mon=0 ... Sun=6. Closed strictly between Friday 17:00 and
    # Sunday 17:00; open every other moment.
    if now.weekday() == 4 and now >= week_close:  # Friday, after the close
        is_open = False
    elif now.weekday() == 5:  # Saturday — always closed
        is_open = False
    elif now.weekday() == 6 and now < week_close:  # Sunday, before the reopen
        is_open = False
    else:
        is_open = True

    if is_open:
        # Mirrors _us_equity_clock_heuristic's convention of reporting the
        # most recent open when already open (callers check is_open first).
        days_since_sunday_open = (now.weekday() + 1) % 7
        next_open = (now - timedelta(days=days_since_sunday_open)).replace(
            hour=17, minute=0, second=0, microsecond=0
        )
    else:
        days_until_sunday = (6 - now.weekday()) % 7
        next_open = (now + timedelta(days=days_until_sunday)).replace(
            hour=17, minute=0, second=0, microsecond=0
        )
        if next_open <= now:
            next_open += timedelta(days=7)
    return {"is_open": is_open, "next_open": next_open}
