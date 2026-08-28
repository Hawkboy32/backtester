"""Single-asset backtest engine — long-only by default, with optional
short-only / long+short modes (see PositionMode below).

Executes trades at the close of the bar on which a signal fires (no
look-ahead: the strategy only sees history up to and including that bar).

SHORT-SELLING SUPPORT (2026-08-08, see CLAUDE_NOTES.txt "Add short-selling
as a real, separately-validated mode"). `shares` becomes negative to
represent an open short — this is deliberately NOT a bolted-on separate code
path: `equity = cash + shares * close` and `Trade.pnl = (exit - entry) *
shares` are BOTH already sign-symmetric (verified: a short's cash proceeds
land in `cash`, a negative `shares` then correctly subtracts a rising
liability / adds a falling one). The only real new logic is which signal is
allowed to OPEN vs CLOSE in which direction — see `position_mode` below.
`PositionMode.LONG_ONLY` (the default) reproduces every existing result
byte-for-byte — nothing about the long-only path changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import pandas as pd

from backtester.conviction import compute_conviction
from backtester.strategy import Bar, Signal, Strategy

# Bump whenever a change affects backtest CORRECTNESS (not cosmetics) — lets
# scan_db.py tag every recorded run with the engine that produced it, so a
# later analysis or roster re-evaluation can tell a pre-fix result from a
# post-fix one instead of trusting insertion order. First real use: the
# 2026-07-27 fix below, where slippage was applied in the same direction for
# both BUY and SELL fills, making a round trip's real transaction cost
# silently near-zero regardless of the slippage_bps setting — every scan run
# before this tag existed was computed under that bug.
ENGINE_VERSION = "2026-08-14-protective-exits"


class PositionMode(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    LONG_SHORT = "long_short"


def _session_window_starts(index: pd.DatetimeIndex, sessions_needed: int) -> list[int]:
    """For every bar position i, the start index of a window covering the
    trailing `sessions_needed` sessions (calendar days) ending at i's own
    session, inclusive. Computed once up front (O(n) total, one pass) so the
    main loop can look up a bar's window start in O(1) instead of
    recomputing session boundaries from scratch on every single bar.
    """
    from backtester.strategies.indicators import session_dates

    dates = session_dates(index)
    is_new_session = dates.ne(dates.shift()).to_numpy()
    session_id = is_new_session.cumsum() - 1  # 0-indexed, increments per calendar day

    first_index_of_session: list[int] = []
    for pos, sid in enumerate(session_id):
        if sid == len(first_index_of_session):
            first_index_of_session.append(pos)

    return [first_index_of_session[max(0, sid - sessions_needed + 1)] for sid in session_id]


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp | None = None
    exit_price: float | None = None
    # Positive = long, NEGATIVE = short (see module docstring — this sign
    # convention is what makes .pnl below correct for both without a branch).
    shares: float = 0.0
    conviction: float | None = None  # [0,1] entry-signal strength (#25); logged only, never sizes
    # What actually closed the trade: None/"signal" = the strategy's own exit
    # signal (the only possibility before protective exits existed), else
    # "stop_loss" / "take_profit" / "max_hold" / "session_end" / "end_of_data".
    # Recorded so a sweep can see WHICH exit did the work rather than only the
    # net result — a stop that never fires and a stop that fires constantly
    # produce very different equity curves for the same parameter.
    exit_reason: str | None = None

    @property
    def pnl(self) -> float | None:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.shares


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    regime_curve: pd.Series | None = None


class BacktestEngine:
    def __init__(
        self,
        starting_cash: float = 100_000.0,
        commission_per_trade: float = 0.0,
        slippage_bps: float = 0.0,
        regime_by_date: dict | None = None,
        blocked_dates: set | None = None,
        position_mode: PositionMode = PositionMode.LONG_ONLY,
        fixed_dollars_per_trade: float | None = None,
        dynamic_size_fn: Callable[[float], float] | None = None,
        stop_loss_pct: float | None = None,
        take_profit_pct: float | None = None,
        trailing_arm_pct: float | None = None,
        trailing_stop_pct: float | None = None,
        max_hold_bars: int | None = None,
        close_at_session_end: bool = False,
        deposits: dict | None = None,
        variable_slippage_fn: Callable[[float, float], float] | None = None,
        regular_hours_only: bool = False,
    ):
        """fixed_dollars_per_trade: when set, every entry spends
        min(cash, fixed_dollars_per_trade) instead of the default all-in
        cash * size_multiplier — matches how a live account configured with
        auto_trader_state.py's account_sizing_overrides (a fixed $/trade,
        not pct_equity) actually gets sized, which this engine had no way to
        reproduce before (added 2026-08-09, found while backtesting a small
        live-pilot account and getting all-in-compounded numbers that
        overstated what a fixed-size account would really do by an order of
        magnitude). None (default) preserves the exact original all-in
        behavior — zero regression for every existing caller.

        dynamic_size_fn: optional callable(current_cash) -> fraction of cash
        to spend on THIS entry (same units as size_multiplier, e.g. 0.02 for
        2%) — recomputed fresh at every single entry against whatever cash
        actually is at that moment, unlike fixed_dollars_per_trade (a fixed
        dollar amount) or size_multiplier (a fixed fraction for the whole
        backtest). Added 2026-08-10 to genuinely backtest
        execution.sliding_pct_equity's real behavior — a small-account
        override whose own rate changes as equity grows — rather than
        approximate it with a single static number. Takes priority over
        both fixed_dollars_per_trade and size_multiplier when set. None
        (default) preserves existing behavior exactly.

        regime_by_date: optional {date: {"regime": "calm"/"normal"/"storm", "size_multiplier": float}},
        e.g. from backtester.volatility.regime_by_date(). When a bar's date has
        an entry: new entries (either direction — see position_mode) are
        blocked while regime == "storm", and the cash committed to a new
        position is scaled by size_multiplier instead of using all available
        cash. Dates with no entry (or when this is None) behave exactly as
        before — all-in, no filter.

        blocked_dates: optional set of datetime.date on which NEW entries are
        suppressed (exits still fire) — e.g. backtester.events
        .blocked_dates_in_range() for known risk-event days like FOMC. The
        proactive complement to the reactive storm-regime block above.

        position_mode: which signal direction(s) may OPEN a new position —
        LONG_ONLY (default, today's exact existing behavior: BUY opens/SELL
        closes only), SHORT_ONLY (SELL opens a short/BUY closes it — mirrors
        LONG_ONLY exactly, just flipped), or LONG_SHORT (either signal may
        open while flat, whichever fires first — still only ONE position
        open at a time, never both directions simultaneously). See the
        module docstring for why this needed no change to the P&L/equity
        math itself, only to which signal is allowed to open/close when.

        PROTECTIVE EXITS (2026-08-14). All four default to off, so every
        existing caller and every previously recorded result is unchanged.
        Until these existed a position could ONLY be closed by the strategy's
        own exit signal, so a thesis that never came good was held
        indefinitely — the live account held DDOG for 43h (-3.6%) and AEP for
        7 days (-6.1%) for exactly that reason.

        stop_loss_pct / take_profit_pct: fraction of the ENTRY price, e.g.
        0.01 = 1%. Checked INTRABAR against each bar's low/high (not its
        close): a resting broker-side stop triggers the moment price touches
        it, and testing against the close alone would silently miss most stop
        hits and overstate results.

        trailing_arm_pct / trailing_stop_pct (2026-08-28): a TRAILING exit,
        distinct from the fixed take_profit_pct above. trailing_arm_pct is
        how far in profit (fraction of entry price) a position must move at
        least once before the trailing exit can fire at all — before that,
        an ordinary post-entry wiggle can never trigger it. trailing_stop_pct
        is the fraction of the PEAK price reached since entry (not the entry
        price — the standard trailing-stop convention) that price may give
        back before it fires. Both None (default) is a zero-regression no-op.
        Setting trailing_stop_pct without trailing_arm_pct raises — a trail
        distance with no arm threshold is ambiguous about when it starts.
        Requires "armed" no separate flag: the peak is monotonic once
        tracked (only ever moves further in the position's favor), so
        whether the arm threshold has been reached is a pure function of the
        current peak vs entry — do not "helpfully" add an armed flag here,
        it would just be redundant state that could drift from the peak it's
        derived from. Added specifically for the case a fixed take_profit_pct
        can't address: a position that recovers into real profit and then
        gives it all back before the strategy's own signal exits (found live
        in the 2026-08-27 loss diagnostic — 6 of 10 recovered-to-entry
        losing trades fell away again for a genuine second leg down, some
        held 700+ minutes). take_profit_pct may still be set alongside
        trailing as a hard ceiling for whatever the trail never catches.

        max_hold_bars: force an exit at the close of the Nth bar after entry,
        whatever the price is doing.

        close_at_session_end: flatten at the close of each session's last bar
        rather than carrying a position overnight.

        Three deliberately CONSERVATIVE modelling choices, because all are
        places a backtest can flatter itself:
          - If a bar's range contains BOTH the stop and the target, OHLC data
            cannot say which was touched first. This assumes the STOP — the
            worse outcome. Assuming the target would inflate every result and
            is the classic way stop/target backtests lie.
          - Same reasoning extended to trailing vs. take_profit_pct: if a bar
            could satisfy both, this assumes TRAILING fired — its exit price
            (peak - trail distance) is necessarily closer to entry (less
            profit) than take_profit_pct's fixed target, so assuming the
            target instead would again be the self-flattering assumption.
          - A bar that GAPS through a level fills at that bar's OPEN, not at
            the level. A stop gapped through fills worse than the stop price,
            which is what actually happens; pretending otherwise would hide
            precisely the overnight gap risk close_at_session_end exists to
            avoid.
        Slippage still applies on top of both, in the same direction as any
        other exit of that side.

        deposits: optional {datetime.date: dollar_amount} - added to cash once,
        on the FIRST bar processed for that date, before that bar's sizing is
        computed (so a deposit is available to size that same day's entries,
        not just the next day's). None (default) is a zero-regression no-op
        for every existing caller. Added 2026-08-15 to replay "what if I kept
        contributing" scenarios; equity_curve/Trade accounting is otherwise
        untouched, but note that total_return derived from equity_curve[0]
        and equity_curve[-1] STOPS being a pure trading-performance number
        once deposits are non-empty - it now also reflects the contributions
        themselves. Callers who need trading performance alone should track
        ending equity minus total deposited, not compute_report's percentage.

        variable_slippage_fn: optional callable(shares, bar_volume) -> slippage
        in bps, called separately for every fill (entries AND exits) with the
        shares actually being traded on that fill and that bar's volume.
        Overrides the flat slippage_bps for that one fill when set; None
        (default) is a zero-regression no-op - every fill uses slippage_bps
        exactly as before. Added 2026-08-16 to answer a specific question a
        flat bps rate cannot: whether a position that has compounded large
        would face real market-impact costs a constant-bps model is blind to.
        For a BUY/SELL that OPENS a new position, the share count used to
        call this is an ESTIMATE (spend / bar close, ignoring slippage
        itself) since the real share count depends circularly on the fill
        price this function is computing - a standard, small approximation
        (slippage is normally a small fraction of price, so estimating share
        count without it first is negligible error). Exits use the real,
        already-known share count directly, no approximation needed.

        regular_hours_only: when True, blocks NEW entries (exits still fire)
        on bars outside 09:30-16:00 America/New_York - the proactive
        complement to blocked_dates/storm-regime blocking, just by
        time-of-day instead of by date. Added 2026-08-16 after the liquidity
        model above showed the worst modeled impact spikes weren't from
        genuinely large orders - they clustered in bars with near-zero
        volume, and cross-checking against trade timestamps showed a large
        fraction of a price-only mean-reversion strategy's entries firing in
        thin pre/after-market bars where almost nobody else is trading.
        NYSE-hours-hardcoded and equity-specific by construction - do not
        set this True for a crypto or forex backtest, both trade on
        different calendars this flag knows nothing about. False (default)
        is a zero-regression no-op. Bar timestamps are assumed UTC-indexed,
        same assumption strategies.vwap_drift_pullback's own wall-clock
        filter already relies on elsewhere in this codebase.
        """
        if stop_loss_pct is not None and stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be a positive fraction of entry price (e.g. 0.01 for 1%)")
        if take_profit_pct is not None and take_profit_pct <= 0:
            raise ValueError("take_profit_pct must be a positive fraction of entry price")
        if trailing_arm_pct is not None and trailing_arm_pct <= 0:
            raise ValueError("trailing_arm_pct must be a positive fraction of entry price")
        if trailing_stop_pct is not None and trailing_stop_pct <= 0:
            raise ValueError("trailing_stop_pct must be a positive fraction of the peak price")
        if trailing_stop_pct is not None and trailing_arm_pct is None:
            raise ValueError("trailing_stop_pct requires trailing_arm_pct — ambiguous where the trail starts otherwise")
        if max_hold_bars is not None and max_hold_bars < 1:
            raise ValueError("max_hold_bars must be >= 1")

        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.trailing_arm_pct = trailing_arm_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.max_hold_bars = max_hold_bars
        self.close_at_session_end = close_at_session_end
        self.starting_cash = starting_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.regime_by_date = regime_by_date
        self.blocked_dates = blocked_dates
        self.position_mode = position_mode
        self.fixed_dollars_per_trade = fixed_dollars_per_trade
        self.dynamic_size_fn = dynamic_size_fn
        self.deposits = deposits
        self.variable_slippage_fn = variable_slippage_fn
        self.regular_hours_only = regular_hours_only

    def _slip_bps(self, shares: float, bar_volume: float) -> float:
        """Slippage (in bps) for a fill of this size on this bar. Falls back
        to the flat self.slippage_bps when variable_slippage_fn is None -
        every existing caller's math is untouched."""
        if self.variable_slippage_fn is None:
            return self.slippage_bps
        return self.variable_slippage_fn(abs(shares), bar_volume)

    def _protective_exit(
        self, entry_price: float, shares: float, current: Bar, bars_held: int, is_session_end: bool,
        peak_price: float | None = None,
    ) -> tuple[float, str] | None:
        """The raw fill price and reason if a protective exit fires on this bar,
        else None. Price is PRE-slippage; the caller applies it directionally.

        Precedence is deliberate: intrabar price levels are checked before the
        end-of-bar time rules, because a resting stop or target triggers the
        instant price touches it — necessarily at or before this bar's close.
        Stop is checked before trailing, and trailing before the fixed target
        — a bar spanning multiple levels always resolves to whichever is
        closest to entry (the conservative outcome; see __init__ for why).

        peak_price: the high-water mark (long) / low-water mark (short) since
        entry, maintained by the caller (run()) every bar a position is open
        — this function stays a pure function of its arguments, no state of
        its own, same as every other check here.
        """
        is_long = shares > 0

        stop_price = target_price = None
        if self.stop_loss_pct is not None:
            stop_price = entry_price * ((1 - self.stop_loss_pct) if is_long else (1 + self.stop_loss_pct))
        if self.take_profit_pct is not None:
            target_price = entry_price * ((1 + self.take_profit_pct) if is_long else (1 - self.take_profit_pct))

        if stop_price is not None:
            if (current.low <= stop_price) if is_long else (current.high >= stop_price):
                # A bar that opened beyond the stop gapped through it overnight;
                # the order fills at the open, which is WORSE than the stop.
                gapped = (current.open <= stop_price) if is_long else (current.open >= stop_price)
                return (current.open if gapped else stop_price), "stop_loss"

        if self.trailing_arm_pct is not None and peak_price is not None:
            armed = (
                (peak_price - entry_price) >= entry_price * self.trailing_arm_pct if is_long
                else (entry_price - peak_price) >= entry_price * self.trailing_arm_pct
            )
            if armed and self.trailing_stop_pct is not None:
                trail_price = (
                    peak_price * (1 - self.trailing_stop_pct) if is_long
                    else peak_price * (1 + self.trailing_stop_pct)
                )
                if (current.low <= trail_price) if is_long else (current.high >= trail_price):
                    gapped = (current.open <= trail_price) if is_long else (current.open >= trail_price)
                    return (current.open if gapped else trail_price), "trailing_stop"

        if target_price is not None:
            if (current.high >= target_price) if is_long else (current.low <= target_price):
                gapped = (current.open >= target_price) if is_long else (current.open <= target_price)
                return (current.open if gapped else target_price), "take_profit"

        if self.max_hold_bars is not None and bars_held >= self.max_hold_bars:
            return current.close, "max_hold"
        if self.close_at_session_end and is_session_end:
            return current.close, "session_end"
        return None

    def run(self, bars: pd.DataFrame, strategy: Strategy) -> BacktestResult:
        if bars.empty:
            raise ValueError("No bars provided to backtest")

        # How much trailing history each bar's on_bar() call actually needs
        # to see, computed once up front — see Lookback's docstring. Both
        # None (the default for any strategy that hasn't opted in) means
        # unbounded: window_start stays 0 for every bar, i.e. the exact same
        # full-history-so-far behavior as before this existed.
        lookback = strategy.required_lookback()
        session_starts: list[int] | None = None
        if lookback.sessions is not None:
            session_starts = _session_window_starts(bars.index, lookback.sessions)

        # True on each bar that is the LAST of its session. The final bar of
        # the data counts (shift(-1) is NaN there, so ne() is True), which is
        # correct: an open position there is liquidated anyway.
        session_end_flags = None
        if self.close_at_session_end:
            from backtester.strategies.indicators import session_dates

            dates = session_dates(bars.index)
            session_end_flags = dates.ne(dates.shift(-1)).to_numpy()

        # True on each bar that falls inside 09:30-16:00 America/New_York -
        # precomputed once (vectorized) rather than converting a timezone on
        # every single bar in the loop below, same reasoning as session_end_flags.
        regular_hours_flags = None
        if self.regular_hours_only:
            et_index = bars.index.tz_convert("America/New_York")
            minutes_of_day = et_index.hour * 60 + et_index.minute
            regular_hours_flags = (minutes_of_day >= 9 * 60 + 30) & (minutes_of_day < 16 * 60)

        cash = self.starting_cash
        shares = 0.0
        entry_index: int | None = None
        open_trade: Trade | None = None
        # High-water mark (long) / low-water mark (short) since entry, for
        # trailing_stop_pct. Monotonic by construction (max/min only ever
        # moves it further in the position's favor), so "armed" needs no
        # separate flag — see _protective_exit's docstring.
        peak_price: float | None = None
        trades: list[Trade] = []
        equity_values: list[float] = []
        regime_values: list[str | None] = [] if self.regime_by_date is not None else None
        last_deposit_date = None

        for i in range(len(bars)):
            if lookback.bars is not None:
                window_start = max(0, i + 1 - lookback.bars)
            elif session_starts is not None:
                window_start = session_starts[i]
            else:
                window_start = 0
            history = bars.iloc[window_start : i + 1]
            row = bars.iloc[i]
            current = Bar(
                timestamp=bars.index[i],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )

            if self.deposits is not None:
                current_date = pd.Timestamp(current.timestamp).date()
                if current_date != last_deposit_date:
                    cash += self.deposits.get(current_date, 0.0)
                    last_deposit_date = current_date

            signal = strategy.on_bar(history, current)
            # Slippage must be DIRECTIONAL — it costs the trader on both sides of a
            # round trip, never nets to ~zero. A BUY fills at a WORSE (higher) price;
            # a SELL fills at a WORSE (lower) price. (Bug found 2026-07-27: a single
            # shared `close * (1 + bps)` fill price for both sides made a round trip's
            # P&L nearly slippage-INVARIANT — the extra cost on entry and the extra
            # "benefit" on exit almost exactly canceled, confirmed with a flat-price
            # repro showing a literal $0.00 slippage cost on a 1% slippage round trip.
            # This silently understated every backtest's real transaction costs since
            # this engine was first written.)
            #
            # Computed lazily per fill, not once per bar, since variable_slippage_fn
            # (added 2026-08-16) needs the actual share count of THIS fill, which
            # differs between an entry, an exit, and a protective exit.

            regime = None
            size_multiplier = 1.0
            if self.regime_by_date is not None:
                info = self.regime_by_date.get(pd.Timestamp(current.timestamp).date())
                if info is not None:
                    regime = info["regime"]
                    size_multiplier = info["size_multiplier"]
                regime_values.append(regime)

            event_blocked = (
                self.blocked_dates is not None
                and pd.Timestamp(current.timestamp).date() in self.blocked_dates
            )
            hours_blocked = regular_hours_flags is not None and not regular_hours_flags[i]

            can_open_long = self.position_mode in (PositionMode.LONG_ONLY, PositionMode.LONG_SHORT)
            can_open_short = self.position_mode in (PositionMode.SHORT_ONLY, PositionMode.LONG_SHORT)
            allow_new_entry = regime != "storm" and not event_blocked and not hours_blocked
            # fixed_dollars_per_trade (when set) overrides the default all-in
            # cash * size_multiplier - capped at available cash so a nearly-
            # exhausted account can't "spend" more than it has. dynamic_size_fn
            # takes priority (recomputed fresh against THIS bar's cash), then
            # fixed_dollars_per_trade, then the plain size_multiplier default.
            if self.dynamic_size_fn is not None:
                spend = cash * self.dynamic_size_fn(cash)
            elif self.fixed_dollars_per_trade is not None:
                spend = min(cash, self.fixed_dollars_per_trade)
            else:
                spend = cash * size_multiplier

            protective = None
            if shares != 0 and open_trade is not None and entry_index is not None:
                # Update this bar's extreme BEFORE checking the trailing
                # condition, so the check sees the same bar's own high/low —
                # matching how stop/target already check THIS bar's touch,
                # not last bar's.
                peak_price = max(peak_price, current.high) if shares > 0 else min(peak_price, current.low)
                protective = self._protective_exit(
                    open_trade.entry_price,
                    shares,
                    current,
                    i - entry_index,
                    bool(session_end_flags[i]) if session_end_flags is not None else False,
                    peak_price,
                )

            if protective is not None:
                # Pre-empts the strategy's own signal for this bar. Note the
                # position is NOT reopened on the same bar even if the signal
                # says to: a real resting stop fills intrabar and could not be
                # followed by a fresh entry at that same bar's close, and
                # allowing it would let a stop sweep manufacture extra round
                # trips that never existed.
                raw_price, reason = protective
                exit_slip = self._slip_bps(shares, current.volume) / 10_000
                if shares > 0:
                    fill = raw_price * (1 - exit_slip)
                    cash += shares * fill - self.commission_per_trade
                else:
                    fill = raw_price * (1 + exit_slip)
                    cash -= abs(shares) * fill + self.commission_per_trade
                open_trade.exit_time = current.timestamp
                open_trade.exit_price = fill
                open_trade.exit_reason = reason
                trades.append(open_trade)
                open_trade = None
                entry_index = None
                peak_price = None
                shares = 0.0
            elif signal is Signal.BUY and shares == 0 and can_open_long and allow_new_entry:
                if spend > self.commission_per_trade:
                    approx_shares = spend / current.close  # ignores slippage - see variable_slippage_fn docstring
                    entry_slip = self._slip_bps(approx_shares, current.volume) / 10_000
                    buy_fill_price = current.close * (1 + entry_slip)
                    shares = (spend - self.commission_per_trade) / buy_fill_price
                    cash -= shares * buy_fill_price + self.commission_per_trade
                    # Conviction is scored at the entry bar and stored for later learning
                    # (#25). It does NOT influence sizing here — spend/shares are unchanged.
                    conviction = compute_conviction(strategy, history, current)
                    open_trade = Trade(
                        entry_time=current.timestamp, entry_price=buy_fill_price,
                        shares=shares, conviction=conviction,
                    )
                    entry_index = i
                    peak_price = buy_fill_price
            elif signal is Signal.SELL and shares == 0 and can_open_short and allow_new_entry:
                # Opens a SHORT — shares goes negative (see module docstring for
                # why equity/pnl need no special-casing for this). Sized the
                # same way a long entry is: the notional "spend" is the short's
                # dollar exposure, not literal cash outlay (a short generates
                # proceeds rather than consuming cash up front).
                if spend > self.commission_per_trade:
                    approx_shares = spend / current.close
                    entry_slip = self._slip_bps(approx_shares, current.volume) / 10_000
                    sell_fill_price = current.close * (1 - entry_slip)
                    shares = -(spend - self.commission_per_trade) / sell_fill_price
                    cash += abs(shares) * sell_fill_price - self.commission_per_trade
                    conviction = compute_conviction(strategy, history, current)
                    open_trade = Trade(
                        entry_time=current.timestamp, entry_price=sell_fill_price,
                        shares=shares, conviction=conviction,
                    )
                    entry_index = i
                    peak_price = sell_fill_price
            elif signal is Signal.SELL and shares > 0:
                # Closes an existing LONG (unchanged from before position_mode existed).
                exit_slip = self._slip_bps(shares, current.volume) / 10_000
                sell_fill_price = current.close * (1 - exit_slip)
                cash += shares * sell_fill_price - self.commission_per_trade
                if open_trade is not None:
                    open_trade.exit_time = current.timestamp
                    open_trade.exit_price = sell_fill_price
                    open_trade.exit_reason = "signal"
                    trades.append(open_trade)
                    open_trade = None
                entry_index = None
                peak_price = None
                shares = 0.0
            elif signal is Signal.BUY and shares < 0:
                # Closes an existing SHORT — buying back to cover costs cash.
                exit_slip = self._slip_bps(shares, current.volume) / 10_000
                buy_fill_price = current.close * (1 + exit_slip)
                cash -= abs(shares) * buy_fill_price + self.commission_per_trade
                if open_trade is not None:
                    open_trade.exit_time = current.timestamp
                    open_trade.exit_price = buy_fill_price
                    open_trade.exit_reason = "signal"
                    trades.append(open_trade)
                    open_trade = None
                entry_index = None
                peak_price = None
                shares = 0.0

            equity = cash + shares * current.close
            # Sanity guard against the accounting corruption seen in scan run 3
            # (silent sub -100% returns from phantom negative shares). With
            # entries blocked when spend <= commission, the one LEGITIMATE way
            # a LONG-side equity dips below zero is a final sell whose proceeds
            # are smaller than the commission — a busted account ends at worst
            # -commission, like a real brokerage charging the fee anyway.
            # Deep negative equity IS legitimate for an open short that's moved
            # heavily against it (unbounded loss potential, unlike a long), so
            # this guard only applies while flat or long (shares >= 0) — a
            # short's own accounting is checked by the symmetric-math tests
            # instead, not this floor. Anything deeper
            # than that is a genuine invariant violation.
            if shares >= 0 and equity < -(self.commission_per_trade + 1e-6):
                raise RuntimeError(
                    f"Equity went to {equity:.2f} at {current.timestamp} — below the "
                    f"-commission floor ({-self.commission_per_trade:.2f}) that a busted "
                    "long-only, no-leverage account can legitimately reach. This indicates "
                    "an accounting bug, not a legitimate loss; treat this result as invalid."
                )
            equity_values.append(equity)

        if open_trade is not None:
            last_row = bars.iloc[-1]
            # A forced liquidation at the end of the window is still a real
            # fill — it must slip the same way an in-loop exit would, not fill
            # at a frictionless raw close (that was a second, smaller instance
            # of the same "final exit gets a free ride" gap fixed above).
            # Direction depends on what's actually open: a long liquidates via
            # a SELL (slips down), a short liquidates via a BUY-to-cover
            # (slips up) — get this backwards and a short's final exit would
            # silently look more profitable than a real cover ever would.
            open_trade.exit_time = bars.index[-1]
            final_slip = self._slip_bps(shares, last_row["volume"]) / 10_000
            if shares > 0:
                open_trade.exit_price = last_row["close"] * (1 - final_slip)
            else:
                open_trade.exit_price = last_row["close"] * (1 + final_slip)
            open_trade.exit_reason = "end_of_data"
            trades.append(open_trade)

        equity_curve = pd.Series(equity_values, index=bars.index, name="equity")
        regime_curve = (
            pd.Series(regime_values, index=bars.index, name="regime")
            if regime_values is not None
            else None
        )
        return BacktestResult(equity_curve=equity_curve, trades=trades, regime_curve=regime_curve)
