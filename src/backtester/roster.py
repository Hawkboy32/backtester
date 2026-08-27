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
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from backtester import ranking
from backtester.auto_trader_state import STATE_DIR, atomic_write_text, from_dict
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
    # True when this entry was paused purely because it fell outside a CAP
    # (top-N / max_per_strategy / max_per_sector) rather than for poor live
    # performance. Those two need opposite handling once the position is
    # flat: a cap-demotion should leave the roster entirely, while a
    # performance-pause stays benched. See release_flat_paused().
    demoted_for_cap: bool = False
    # Set when a stale performance-pause is auto-released (see
    # pause_release_days): the combo's live trade count AT that moment. The
    # losing-streak rule is skipped while num_trades hasn't moved past this,
    # so the very streak that caused the pause can't immediately re-trip it —
    # otherwise release would be a no-op. Every OTHER demotion rule still
    # applies, and this self-clears as soon as one new trade closes.
    grace_after_trades: int | None = None


@dataclass
class RosterConfig:
    roster_size: int = 4
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
    # What the overnight auto-rescan (auto_trader.py, after market close) scans
    # when it finds a roster gap — explicit and user-set, NOT inferred from
    # scan_history.db's own "most recent run", which isn't reliable for this:
    # confirmed live that the literal most-recent row can be a small ad-hoc
    # scan from unrelated dev/debug work (found one recorded as "S&P 500" with
    # only 6 tickers), which would silently replay the wrong universe. Defaults
    # match what actually built the real roster (DDOG/MPWR/SWK).
    rescan_universes: list = field(default_factory=lambda: ["S&P 500", "Nasdaq-100 (US Tech 100)"])
    rescan_strategy_names: list = field(default_factory=lambda: ["VWAP Mean Reversion", "Bollinger Mean Reversion"])
    rescan_window_days: int = 30
    # Minimum days between overnight rescans even when there's no roster gap —
    # catches a decaying edge via fresh re-ranking before live performance
    # forces a demotion, not just reacting to one after the fact.
    review_cadence_days: int = 7
    # Days a PERFORMANCE-paused combo stays benched before it's automatically
    # given another chance. 0 disables (pauses become permanent again).
    #
    # Fixes a real deadlock found 2026-08-13: a paused combo can only EXIT,
    # never open, so it can never place another trade — which means
    # current_losing_streak can never fall back below the threshold that
    # paused it. MPWR sat permanently benched on a 3-loss streak despite
    # +$22 net and a 70% win rate, and no amount of re-evaluating could
    # release it. Re-admission is gated on a free roster slot and comes with
    # a one-trade grace on the streak rule (see grace_after_trades), so the
    # combo genuinely re-proves itself rather than being re-paused instantly
    # by the same stale history.
    pause_release_days: int = 5


@dataclass
class RosterState:
    entries: list[RosterEntry] = field(default_factory=list)
    config: RosterConfig = field(default_factory=RosterConfig)


def load_roster() -> RosterState:
    if not ROSTER_PATH.exists():
        return RosterState()
    try:
        data = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
        entries = [from_dict(RosterEntry, e) for e in data.get("entries", [])]
        config = from_dict(RosterConfig, data.get("config", {}))
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


def _demotion_reason(stats, config: RosterConfig, grace_after_trades: int | None = None) -> str | None:
    """A human-readable reason to pause/reject, or None if the combo is fine.
    Returns None (never blocks) until enough live trades exist to judge it —
    backtest score alone drives an initial promotion.

    grace_after_trades: set on a combo just auto-released from a stale pause
    (see pause_release_days). While its trade count hasn't moved past that
    mark, EVERY demotion rule below is skipped, not just the losing-streak
    one — otherwise the identical frozen numbers that caused the pause would
    re-trip it on the very next check and release would achieve nothing.

    2026-08-27 fix: this used to gate the losing-streak rule ALONE, leaving
    the win-rate and cumulative-P&L-floor rules unconditional. Those are just
    as frozen as the streak while no new trade has closed, so a combo paused
    for either one was structurally unable to ever be released — confirmed
    live: Q/VWAP Mean Reversion was manually released three times and
    re-paused within seconds to minutes each time, always citing the exact
    same cumulative-P&L figure, because nothing had traded in between to
    change it and grace didn't cover that rule. The grace expires by itself
    the moment one new trade closes, at which point every rule re-engages on
    genuinely fresh data.
    """
    if stats.num_trades < config.min_live_trades:
        return None
    in_grace = grace_after_trades is not None and stats.num_trades <= grace_after_trades
    if in_grace:
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


def release_flat_paused(state: RosterState, held_tickers: set[str]) -> tuple[RosterState, list[str]]:
    """Retire cap-demoted entries once their position is actually closed.

    The other half of the fix in evaluate_roster: a combo demoted purely for
    falling outside a cap is kept as "paused" so it can still EXIT, but it
    must not stay there forever — every paused entry costs a data fetch and a
    signal evaluation on every poll cycle, so left unchecked they'd
    accumulate indefinitely. Once nothing holds it, it retires to "candidate"
    and drops out of the trading loop.

    Only touches entries with demoted_for_cap=True. A performance-pause
    (losing streak, win-rate floor) is deliberately left benched regardless of
    whether it holds anything — that's a judgement about the strategy, not a
    position-lifecycle concern.

    Returns (new_state, retired_labels) so the caller can log what changed.
    """
    retired: list[str] = []
    new_entries = []
    for e in state.entries:
        if e.status == "paused" and e.demoted_for_cap and e.ticker not in held_tickers:
            retired.append(f"{e.ticker}/{e.strategy_name}")
            new_entries.append(replace(
                e, status="candidate", paused_at=None, pause_reason=None, demoted_for_cap=False,
            ))
        else:
            new_entries.append(e)
    return RosterState(entries=new_entries, config=state.config), retired


def apply_demotion_checks(
    current_roster: RosterState,
    live_perf_fn: Callable[[str, str], object],
    config: RosterConfig | None = None,
) -> RosterState:
    """Cheap path: no re-ranking, no scan data needed. Safe to call every
    auto_trader.py poll cycle. Moves active -> paused on a live-performance
    breach, and (since 2026-08-13) paused -> active for a PERFORMANCE pause
    that has gone stale — see pause_release_days for the deadlock that
    required. Never promotes a candidate; that's still evaluate_roster's job.
    """
    config = config or current_roster.config
    now = datetime.now(timezone.utc)
    active_count = sum(1 for e in current_roster.entries if e.status == "active")

    new_entries = []
    for entry in current_roster.entries:
        stats = live_perf_fn(entry.ticker, entry.strategy_name)
        entry.live_stats = asdict(stats)

        if entry.status == "active":
            reason = _demotion_reason(stats, config, entry.grace_after_trades)
            if reason is not None:
                entry.status = "paused"
                entry.paused_at = now.isoformat()
                entry.pause_reason = reason
                entry.grace_after_trades = None  # a fresh pause starts from scratch
                append_event(entry.ticker, entry.strategy_name, "paused", reason)
            elif entry.grace_after_trades is not None and stats.num_trades > entry.grace_after_trades:
                entry.grace_after_trades = None  # traded again and survived — grace done

        elif (
            entry.status == "paused"
            and not entry.demoted_for_cap          # cap demotions retire via release_flat_paused
            and config.pause_release_days > 0
            and active_count < config.roster_size  # never exceed the cap by releasing
            and entry.paused_at
        ):
            try:
                benched_days = (now - datetime.fromisoformat(entry.paused_at)).total_seconds() / 86400
            except ValueError:
                benched_days = 0.0
            if benched_days >= config.pause_release_days:
                entry.status = "active"
                entry.pause_reason = None
                entry.paused_at = None
                # Skip the streak rule until it actually trades again, so the
                # frozen streak that paused it can't instantly re-pause it.
                entry.grace_after_trades = stats.num_trades
                active_count += 1
                append_event(
                    entry.ticker, entry.strategy_name, "released",
                    f"auto-released after {benched_days:.1f} days benched (>= {config.pause_release_days})",
                )
        new_entries.append(entry)

    return RosterState(entries=new_entries, config=config)


def evaluate_roster(
    scan_rows: list[ScanResultRow],
    live_perf_fn: Callable[[str, str], object],
    current_roster: RosterState,
    config: RosterConfig | None = None,
    score_fn: ScoreFn = ranking.rank_combos,
    dry_run: bool = False,
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

    dry_run: when True, computes the proposed RosterState WITHOUT writing any
    append_event() audit entries — used by compute_recommendation() to preview
    what evaluate_roster WOULD do without misleadingly recording "promoted"/
    "paused"/"demoted" events for a change nobody has actually approved yet.
    The returned RosterState is identical either way; only the audit-log side
    effect is suppressed.
    """
    config = config or current_roster.config
    _log_event = (lambda *a, **kw: None) if dry_run else append_event
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
                    _log_event(
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
        # Carry any active streak-grace through a re-evaluation, so a combo
        # just auto-released from a stale pause isn't immediately re-paused
        # here by the same frozen streak it was released from.
        grace = existing.grace_after_trades if existing else None
        reason = _demotion_reason(stats, config, grace)

        if reason is not None:
            # Preserve the ORIGINAL paused_at when it was already paused —
            # this timestamp is the bench clock that pause_release_days counts
            # from, and re-stamping it on every re-evaluation would reset that
            # clock. A combo whose stale streak makes `reason` fire forever
            # would then never reach the release threshold, exactly the
            # deadlock pause_release_days exists to break (and a weekly
            # auto-review would silently re-arm it every 7 days).
            already_paused = existing is not None and existing.status == "paused" and existing.paused_at
            entry = RosterEntry(
                ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
                status="paused",
                promoted_at=existing.promoted_at if existing else None,
                paused_at=existing.paused_at if already_paused else datetime.now(timezone.utc).isoformat(),
                pause_reason=reason,
                backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
                grace_after_trades=grace,
            )
            if was_active:
                _log_event(ticker, strategy_name, "paused", reason)
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
            # A combo that was ACTIVE may still be holding a real position, so
            # it becomes "paused" — the one status _resolve_targets still
            # trades, exit-only — rather than "candidate", which it stops
            # trading entirely. Dropping a position-holding combo straight to
            # candidate stranded it: nothing could ever close it again.
            # Happened for real on 2026-08-12 (DDOG, left open across three
            # accounts, only noticed a day later). release_flat_paused() below
            # completes the lifecycle by retiring it to candidate once the
            # position is actually gone, so paused entries can't pile up.
            demoting_held = bool(was_active)
            entry = RosterEntry(
                ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
                status="paused" if demoting_held else "candidate",
                promoted_at=existing.promoted_at if existing else None,
                paused_at=datetime.now(timezone.utc).isoformat() if demoting_held else None,
                pause_reason=cap_reason if demoting_held else None,
                backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
                demoted_for_cap=demoting_held,
            )
            if was_active:
                _log_event(ticker, strategy_name, "demoted", cap_reason)
            new_entries.append(entry)
            seen_keys.add(key)
            continue

        entry = RosterEntry(
            ticker=ticker, strategy_name=strategy_name, params=params_lookup.get(key, {}),
            status="active",
            promoted_at=existing.promoted_at if was_active else datetime.now(timezone.utc).isoformat(),
            backtest_score=score, live_stats=asdict(stats), efficiency_ratio=er,
            grace_after_trades=grace,  # preserved, so a re-eval doesn't cancel an in-flight grace
        )
        if not was_active:
            _log_event(ticker, strategy_name, "promoted", f"backtest score {score:.3f}")
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


RECOMMENDATION_PATH = STATE_DIR / "roster_recommendation.json"


@dataclass
class PendingRecommendation:
    """A computed-but-not-yet-applied evaluate_roster() result — the roster
    stays exactly as apply_demotion_checks left it until a human explicitly
    approves this (dashboard or mobile), same "nothing changes silently"
    principle the roster has always followed, just detected and computed
    automatically instead of requiring someone to remember to check."""
    computed_at: str
    scan_run_id: int
    num_scan_results: int
    summary: list[str]  # human-readable per-combo diff lines, e.g. "MPWR/VWAP Mean Reversion: paused -> active"
    proposed_state: RosterState


def load_pending() -> PendingRecommendation | None:
    if not RECOMMENDATION_PATH.exists():
        return None
    try:
        data = json.loads(RECOMMENDATION_PATH.read_text(encoding="utf-8"))
        proposed = data["proposed_state"]
        entries = [from_dict(RosterEntry, e) for e in proposed.get("entries", [])]
        config = from_dict(RosterConfig, proposed.get("config", {}))
        return PendingRecommendation(
            computed_at=data["computed_at"],
            scan_run_id=data["scan_run_id"],
            num_scan_results=data["num_scan_results"],
            summary=data["summary"],
            proposed_state=RosterState(entries=entries, config=config),
        )
    except Exception:
        return None


def save_pending(rec: PendingRecommendation) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "computed_at": rec.computed_at,
        "scan_run_id": rec.scan_run_id,
        "num_scan_results": rec.num_scan_results,
        "summary": rec.summary,
        "proposed_state": {
            "entries": [asdict(e) for e in rec.proposed_state.entries],
            "config": asdict(rec.proposed_state.config),
        },
    }
    atomic_write_text(RECOMMENDATION_PATH, json.dumps(data, indent=2))


def clear_pending() -> None:
    if RECOMMENDATION_PATH.exists():
        RECOMMENDATION_PATH.unlink()


def compute_recommendation(
    current_roster: RosterState,
    scan_rows: list[ScanResultRow],
    run_id: int,
    live_perf_fn: Callable[[str, str], object],
) -> PendingRecommendation | None:
    """Preview what evaluate_roster WOULD do against fresh scan_rows, without
    saving it as the live roster or recording audit events for a change
    nobody has approved yet (see evaluate_roster's dry_run param). Returns
    None if the proposed state is identical to the current one — no point
    surfacing a no-op recommendation.
    """
    proposed = evaluate_roster(scan_rows, live_perf_fn, current_roster, dry_run=True)

    current_by_key = {(e.ticker, e.strategy_name): e.status for e in current_roster.entries}
    proposed_by_key = {(e.ticker, e.strategy_name): e.status for e in proposed.entries}

    summary: list[str] = []
    for key in sorted(set(current_by_key) | set(proposed_by_key)):
        old_status = current_by_key.get(key, "candidate")
        new_status = proposed_by_key.get(key, "candidate")
        if old_status != new_status:
            ticker, strategy_name = key
            summary.append(f"{ticker}/{strategy_name}: {old_status} -> {new_status}")

    if not summary:
        return None

    return PendingRecommendation(
        computed_at=datetime.now(timezone.utc).isoformat(),
        scan_run_id=run_id,
        num_scan_results=len(scan_rows),
        summary=summary,
        proposed_state=proposed,
    )
