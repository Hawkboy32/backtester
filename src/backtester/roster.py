"""Promotion/demotion engine for the adaptive Auto Trading roster.

Phase 1 of "learn from mistakes": transparent, rule-based — explicitly NOT a
trained model, same spirit as ranking.py's own "not a trained model" scoring.
Backtest performance (via ranking.rank_combos, reused unmodified) decides
which (ticker, strategy) combos get a chance; real live/paper trade outcomes
(via live_trades.recent_performance) decide whether a combo keeps trading —
a combo can have a great backtest score and still get paused if it's actually
losing money right now.

The `score_fn` parameter on evaluate_roster() is the deliberate seam for a
future ML meta-model: it must match ranking.rank_combos's own contract
(list[ScanResultRow], weights) -> DataFrame with a `score` column. A future
model slots in here without touching promotion, demotion, or persistence.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from backtester import ranking
from backtester.auto_trader_state import STATE_DIR, atomic_write_text
from backtester.metrics import classify_ticker_regime
from backtester.scanner import ScanResultRow
from backtester.strategies import strategy_regime
from backtester.universe import sector_for_ticker

ROSTER_PATH = STATE_DIR / "roster.json"
EVENTS_PATH = STATE_DIR / "roster_events.jsonl"

ScoreFn = Callable[[list[ScanResultRow], dict | None], pd.DataFrame]


@dataclass
class RosterEntry:
    ticker: str
    strategy_name: str
    params: dict = field(default_factory=dict)
    status: str = "candidate"  # "active" | "paused" | "candidate"
    promoted_at: str | None = None
    paused_at: str | None = None
    pause_reason: str | None = None
    backtest_score: float = 0.0
    live_stats: dict | None = None
    # Ticker's efficiency ratio from the scan this entry was last evaluated
    # against (see metrics.efficiency_ratio) — display/match context only.
    efficiency_ratio: float | None = None


@dataclass
class RosterConfig:
    roster_size: int = 5
    min_live_trades: int = 5
    losing_streak_threshold: int = 5
    win_rate_floor: float = 0.30
    cum_pnl_floor: float = 0.0
    max_pnl_drawdown_floor: float = 500.0
    weights: dict = field(default_factory=lambda: dict(ranking.DEFAULT_WEIGHTS))
    # Opt-in: only promote combos whose strategy regime tag (trend/range, see
    # STRATEGY_REGISTRY) matches the ticker's measured behaviour over the scan
    # window (efficiency ratio -> metrics.classify_ticker_regime). "either" on
    # either side never blocks. Off by default — measure-first, like conviction.
    regime_match_only: bool = False
    # Diversification caps on a single re-evaluation's ACTIVE promotions. 0 =
    # unlimited. Without these the roster happily fills every slot with one
    # strategy (the live roster was 3/3 Bollinger Mean Reversion), so a single
    # broken edge takes down the whole book at once. Tickers whose sector is
    # unknown are never blocked by the sector cap.
    max_per_strategy: int = 2
    max_per_sector: int = 2


@dataclass
class RosterState:
    entries: list[RosterEntry] = field(default_factory=list)
    config: RosterConfig = field(default_factory=RosterConfig)


def load_roster() -> RosterState:
    if not ROSTER_PATH.exists():
        return RosterState()
    try:
        data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
        entries = [RosterEntry(**e) for e in data.get("entries", [])]
        config = RosterConfig(**data.get("config", {}))
        return RosterState(entries=entries, config=config)
    except Exception:
        return RosterState()


def save_roster(state: RosterState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "entries": [asdict(e) for e in state.entries],
        "config": asdict(state.config),
    }
    atomic_write_text(ROSTER_PATH, json.dumps(data, indent=2))


def append_event(ticker: str, strategy_name: str, action: str, reason: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "strategy_name": strategy_name,
        "action": action,
        "reason": reason,
    }
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_events(limit: int = 200) -> list[dict]:
    if not EVENTS_PATH.exists():
        return []
    events = []
    with EVENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    events.reverse()  # most recent first
    return events[:limit]


def _demotion_reason(stats, config: RosterConfig) -> str | None:
    """A human-readable reason to pause/reject, or None if the combo is fine.
    Returns None (never blocks) until enough live trades exist to judge it —
    backtest score alone drives an initial promotion.
    """
    if stats.num_trades < config.min_live_trades:
        return None
    if stats.current_losing_streak >= config.losing_streak_threshold:
        return f"{stats.current_losing_streak} consecutive live losses (threshold {config.losing_streak_threshold})"
    if stats.win_rate is not None and stats.win_rate < config.win_rate_floor:
        return f"live win rate {stats.win_rate:.0%} below floor {config.win_rate_floor:.0%}"
    if stats.total_pnl < config.cum_pnl_floor:
        return f"live cumulative P&L ${stats.total_pnl:.2f} below floor ${config.cum_pnl_floor:.2f}"
    if stats.max_pnl_drawdown > config.max_pnl_drawdown_floor:
        return f"live drawdown ${stats.max_pnl_drawdown:.2f} exceeds floor ${config.max_pnl_drawdown_floor:.2f}"
    return None


def apply_demotion_checks(
    current_roster: RosterState,
    live_perf_fn: Callable[[str, str], object],
    config: RosterConfig | None = None,
) -> RosterState:
    """Cheap path: no re-ranking, no scan data needed. Safe to call every
    auto_trader.py poll cycle. Only ever moves active -> paused; never
    promotes or un-pauses (that only happens in evaluate_roster, and only
    when a combo's CURRENT live stats no longer trigger a demotion reason).
    """
    config = config or current_roster.config
    new_entries = []
    for entry in current_roster.entries:
        stats = live_perf_fn(entry.ticker, entry.strategy_name)
        entry.live_stats = asdict(stats)

        if entry.status == "active":
            reason = _demotion_reason(stats, config)
            if reason is not None:
                entry.status = "paused"
                entry.paused_at = datetime.now(timezone.utc).isoformat()
                entry.pause_reason = reason
                append_event(entry.ticker, entry.strategy_name, "paused", reason)
        new_entries.append(entry)

    return RosterState(entries=new_entries, config=config)


def evaluate_roster(
    scan_rows: list[ScanResultRow],
    live_perf_fn: Callable[[str, str], object],
    current_roster: RosterState,
    config: RosterConfig | None = None,
    score_fn: ScoreFn = ranking.rank_combos,
) -> RosterState:
    """Expensive path: re-rank via score_fn (default ranking.rank_combos,
    reused unmodified) then walk the ranked list best-to-worst, promoting up
    to `roster_size` combos into "active" — skipping any ticker already
    claimed by a higher-scored combo (at most one active strategy per ticker;
    position gating in this codebase is ticker-level, not per-strategy-lot),
    and skipping (marking "paused" instead) any combo whose CURRENT live
    track record already fails _demotion_reason, regardless of backtest
    score or whether it was active before. A combo paused in an earlier
    evaluation CAN be promoted again here if its live stats no longer trigger
    a demotion reason — apply_demotion_checks alone never un-pauses, only a
    fresh evaluate_roster call reconsiders paused combos, and only on their
    current merits.

    Call manually (a dashboard button) or right after a Scanner run — never
    from inside auto_trader.py's poll loop; scanning is expensive.
    """
    config = config or current_roster.config
    params_lookup = {(r.ticker, r.strategy_name): r.params for r in scan_rows}
    existing_by_key = {(e.ticker, e.strategy_name): e for e in current_roster.entries}

    ranked = score_fn(scan_rows, config.weights)
    if ranked.empty:
        return current_roster

    claimed_tickers: set[str] = set()
    active_count = 0
    per_strategy: dict[str, int] = {}
    per_sector: dict[str, int] = {}
    new_entries: list[RosterEntry] = []
    seen_keys: set[tuple[str, str]] = set()

    for _, row in ranked.iterrows():
        ticker, strategy_name, score = row["ticker"], row["strategy_name"], float(row["score"])
        key = (ticker, strategy_name)
        if ticker in claimed_tickers:
            continue  # a higher-scored combo already claimed this ticker

        # Ticker behaviour over the scanned window — recorded on every entry for
        # display, and (only when regime_match_only) used to gate promotion.
        raw_er = row.get("efficiency_ratio")
        er = None if raw_er is None or pd.isna(raw_er) else float(raw_er)

        if config.regime_match_only:
            ticker_regime = classify_ticker_regime(er)
            strat_regime = strategy_regime(strategy_name)
            if "either" not in (ticker_regime, strat_regime) and ticker_regime != strat_regime:
                # Mismatch (e.g. a range strategy on a trending ticker): not
                # promotable this pass. Kept as a candidate for audit; an
                # existing active entry gets a visible demotion event.
                existing = existing_by_key.get(key)
                was_active = existing is not None and existing.status == "active"
                entry = RosterEntry(
                    ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
                    status="candidate", promoted_at=existing.promoted_at if existing else None,
                    backtest_score=score, efficiency_ratio=er,
                )
                if was_active:
                    append_event(
                        ticker, strategy_name, "demoted",
                        f"regime mismatch: {strategy_name} is a {strat_regime} strategy but "
                        f"{ticker} measured {ticker_regime} (ER {er:.2f}) over the scan window",
                    )
                new_entries.append(entry)
                seen_keys.add(key)
                continue

        existing = existing_by_key.get(key)
        was_active = existing is not None and existing.status == "active"
        stats = live_perf_fn(ticker, strategy_name)
        reason = _demotion_reason(stats, config)

        if reason is not None:
            entry = RosterEntry(
                ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
                status="paused",
                promoted_at=existing.promoted_at if existing else None,
                paused_at=datetime.now(timezone.utc).isoformat(), pause_reason=reason,
                backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
            )
            if was_active:
                append_event(ticker, strategy_name, "paused", reason)
            new_entries.append(entry)
            seen_keys.add(key)
            continue

        # Diversification: don't let one strategy (or one sector) own the whole
        # roster, however good its backtest scores look.
        sector = sector_for_ticker(ticker)
        cap_reason = None
        if active_count >= config.roster_size:
            cap_reason = f"no longer in top {config.roster_size} by backtest score"
        elif config.max_per_strategy and per_strategy.get(strategy_name, 0) >= config.max_per_strategy:
            cap_reason = (
                f"diversification: already {config.max_per_strategy} active combo(s) using "
                f"{strategy_name}"
            )
        elif config.max_per_sector and sector and per_sector.get(sector, 0) >= config.max_per_sector:
            cap_reason = (
                f"diversification: already {config.max_per_sector} active combo(s) in the "
                f"{sector} sector"
            )

        if cap_reason is not None:
            entry = RosterEntry(
                ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
                status="candidate", promoted_at=existing.promoted_at if existing else None,
                backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
            )
            if was_active:
                append_event(ticker, strategy_name, "demoted", cap_reason)
            new_entries.append(entry)
            seen_keys.add(key)
            continue

        entry = RosterEntry(
            ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
            status="active",
            promoted_at=existing.promoted_at if was_active else datetime.now(timezone.utc).isoformat(),
            backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
        )
        if not was_active:
            append_event(ticker, strategy_name, "promoted", f"backtest score {score:.3f}")
        new_entries.append(entry)
        claimed_tickers.add(ticker)
        active_count += 1
        per_strategy[strategy_name] = per_strategy.get(strategy_name, 0) + 1
        if sector:
            per_sector[sector] = per_sector.get(sector, 0) + 1
        seen_keys.add(key)

    # Entries that used to exist but weren't reconsidered this pass (e.g. the
    # ticker dropped out of the scan universe) keep their last-known status
    # for audit/display rather than silently vanishing.
    for key, existing in existing_by_key.items():
        if key not in seen_keys:
            new_entries.append(existing)

    return RosterState(entries=new_entries, config=config)
