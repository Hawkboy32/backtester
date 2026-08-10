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
ENGINE_VERSION = "2026-08-08-position-mode-shorting"


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
        """
        self.starting_cash = starting_cash
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps
        self.regime_by_date = regime_by_date
        self.blocked_dates = blocked_dates
        self.position_mode = position_mode
        self.fixed_dollars_per_trade = fixed_dollars_per_trade
        self.dynamic_size_fn = dynamic_size_fn

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

        cash = self.starting_cash
        shares = 0.0
        open_trade: Trade | None = None
        trades: list[Trade] = []
        equity_values: list[float] = []
        regime_values: list[str | None] = [] if self.regime_by_date is not None else None

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
            slip = self.slippage_bps / 10_000
            buy_fill_price = current.close * (1 + slip)
            sell_fill_price = current.close * (1 - slip)

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

            can_open_long = self.position_mode in (PositionMode.LONG_ONLY, PositionMode.LONG_SHORT)
            can_open_short = self.position_mode in (PositionMode.SHORT_ONLY, PositionMode.LONG_SHORT)
            allow_new_entry = regime != "storm" and not event_blocked
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

            if signal is Signal.BUY and shares == 0 and can_open_long and allow_new_entry:
                if spend > self.commission_per_trade:
                    shares = (spend - self.commission_per_trade) / buy_fill_price
                    cash -= shares * buy_fill_price + self.commission_per_trade
                    # Conviction is scored at the entry bar and stored for later learning
                    # (#25). It does NOT influence sizing here — spend/shares are unchanged.
                    conviction = compute_conviction(strategy, history, current)
                    open_trade = Trade(
                        entry_time=current.timestamp, entry_price=buy_fill_price,
                        shares=shares, conviction=conviction,
                    )
            elif signal is Signal.SELL and shares == 0 and can_open_short and allow_new_entry:
                # Opens a SHORT — shares goes negative (see module docstring for
                # why equity/pnl need no special-casing for this). Sized the
                # same way a long entry is: the notional "spend" is the short's
                # dollar exposure, not literal cash outlay (a short generates
                # proceeds rather than consuming cash up front).
                if spend > self.commission_per_trade:
                    shares = -(spend - self.commission_per_trade) / sell_fill_price
                    cash += abs(shares) * sell_fill_price - self.commission_per_trade
                    conviction = compute_conviction(strategy, history, current)
                    open_trade = Trade(
                        entry_time=current.timestamp, entry_price=sell_fill_price,
                        shares=shares, conviction=conviction,
                    )
            elif signal is Signal.SELL and shares > 0:
                # Closes an existing LONG (unchanged from before position_mode existed).
                cash += shares * sell_fill_price - self.commission_per_trade
                if open_trade is not None:
                    open_trade.exit_time = current.timestamp
                    open_trade.exit_price = sell_fill_price
                    trades.append(open_trade)
                    open_trade = None
                shares = 0.0
            elif signal is Signal.BUY and shares < 0:
                # Closes an existing SHORT — buying back to cover costs cash.
                cash -= abs(shares) * buy_fill_price + self.commission_per_trade
                if open_trade is not None:
                    open_trade.exit_time = current.timestamp
                    open_trade.exit_price = buy_fill_price
                    trades.append(open_trade)
                    open_trade = None
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
            if shares > 0:
                open_trade.exit_price = last_row["close"] * (1 - self.slippage_bps / 10_000)
            else:
                open_trade.exit_price = last_row["close"] * (1 + self.slippage_bps / 10_000)
            trades.append(open_trade)

        equity_curve = pd.Series(equity_values, index=bars.index, name="equity")
        regime_curve = (
            pd.Series(regime_values, index=bars.index, name="regime")
            if regime_values is not None
            else None
        )
        return BacktestResult(equity_curve=equity_curve, trades=trades, regime_curve=regime_curve)
