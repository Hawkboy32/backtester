"""On-demand health check for the auto-trader: run this any time trade volume
looks suspiciously low and want to know whether that's a quiet market or a
real problem.

Entirely read-only - never places an order, never mutates roster/control/
account-risk state (deliberately does NOT call roster.apply_demotion_checks,
unlike auto_trader.py's own _resolve_targets, so running this repeatedly
during the day can't interact with the live bot's own state). Safe to run
while the real auto-trader process is running.

Usage: python diagnose_bot.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from backtester import accounts as accounts_module, events, roster  # noqa: E402
from backtester.auto_trader_state import load_control, load_status  # noqa: E402
from backtester.conviction import compute_conviction  # noqa: E402
from backtester.data import PolygonClient  # noqa: E402
from backtester.execution_log import LOG_PATH as EXECUTION_LOG_PATH  # noqa: E402
from backtester.strategies import STRATEGY_REGISTRY, build_strategy  # noqa: E402
from backtester.strategy import Bar  # noqa: E402

LOOKBACK_DAYS = 90  # mirrors auto_trader.py's own LOOKBACK_DAYS


def _hr(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def _is_pid_alive(pid: int) -> bool:
    """Real OS-level check, unlike auto_trader._another_instance_alive's
    heartbeat-age check - deliberately independent so this script can tell
    the two apart (see PENDING IDEAS in CLAUDE_NOTES.txt)."""
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(pid) in out.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def check_process() -> None:
    _hr("PROCESS & CONTROL STATE")
    status = load_status()
    control = load_control()

    real_alive = _is_pid_alive(status.pid) if status.pid else False
    hb_age = None
    if status.last_heartbeat:
        try:
            hb_age = (
                datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)
            ).total_seconds()
        except ValueError:
            pass

    print(f"status.json says running={status.running}, pid={status.pid}")
    print(f"Real OS check: pid {status.pid} is {'ALIVE' if real_alive else 'NOT RUNNING'}")
    if status.running and not real_alive:
        print(
            "  !! status.json claims the bot is running but the PID doesn't exist on this "
            "machine. This is the known singleton-guard gap (it only checks heartbeat age, "
            "not real liveness) - the bot is almost certainly dead and nothing will trade "
            "until it's restarted."
        )
    if hb_age is not None:
        stale_after = max(300, 3 * control.poll_interval_seconds)
        flag = "STALE" if hb_age > stale_after else "fresh"
        print(f"Last heartbeat: {hb_age:.0f}s ago ({flag}, stale threshold is {stale_after}s)")

    print(f"control.enabled={control.enabled}, killed={control.killed}, allow_live={control.allow_live}")
    if not control.enabled or control.killed:
        print("  !! Bot is disarmed or killed - this alone fully explains zero trades.")
    print(f"trades_today={status.trades_today} (date={status.trades_date}), max_trades_per_day={control.max_trades_per_day}")
    print(f"last_signal: {status.last_signal!r}")
    print(f"last_error:  {status.last_error!r}")
    if status.last_error and ("1225" in status.last_error or "refused" in status.last_error.lower()):
        print(
            "  hint: WinError 1225 / connection refused on a broker check usually means "
            "IB Gateway/TWS isn't running or isn't logged in on the configured host:port - "
            "not a bot bug. Check the IBKR desktop app is up and logged in."
        )


def check_event_calendar() -> None:
    _hr("EVENT CALENDAR COVERAGE")
    today = datetime.now(timezone.utc).date()
    days_left = (events.CALENDAR_COVERS_THROUGH - today).days
    print(f"FOMC calendar covers through {events.CALENDAR_COVERS_THROUGH} ({days_left} days from today)")
    if days_left < 0:
        print(
            "  !! EXPIRED. control.block_event_days will hard-error every cycle and status.last_error "
            "will read '...update backtester/src/backtester/events.py...' - this blocks the WHOLE "
            "cycle's BUY-blocking logic being evaluated correctly. Add next year's FOMC dates to "
            "src/backtester/events.py."
        )
    elif days_left < 30:
        print(f"  Heads up: expires in {days_left} days - add the next FOMC dates soon.")


def check_accounts(account_ids: list[str]) -> None:
    _hr("BROKER ACCOUNT CONNECTIVITY")
    if not account_ids:
        print("No accounts configured in control.json / extra_targets.")
        return
    broker_accounts = accounts_module.build_broker_accounts(account_ids)
    by_id = {b.account_id: b for b in broker_accounts}
    for aid in account_ids:
        b = by_id.get(aid)
        if b is None:
            print(f"[{aid}] could not be built (missing from broker_accounts.json?)")
            continue
        print(f"\n{b.nickname} ({aid[:8]}...):")

        try:
            clock = b.get_market_clock()
            if clock is None:
                print("  market: no clock (24/7 asset) - always open")
            elif clock["is_open"]:
                print("  market: OPEN")
            else:
                nxt = clock.get("next_open")
                print(f"  market: CLOSED (next open {nxt:%A %H:%M} ET)" if nxt else "  market: CLOSED")
        except Exception as e:  # noqa: BLE001
            print(f"  market clock check FAILED: {e}")

        try:
            snap = b.get_account_snapshot()
            print(f"  account snapshot OK: equity={snap.equity}")
        except Exception as e:  # noqa: BLE001
            print(f"  account snapshot FAILED: {e}")

        try:
            positions = b.get_positions()
            print(f"  positions OK: {len(positions)} open ({', '.join(p.ticker for p in positions) or 'none'})")
        except Exception as e:  # noqa: BLE001
            print(f"  positions check FAILED: {e}")


def check_risk_guards(account_ids: list[str]) -> None:
    _hr("RISK GUARDS")
    risk_path = Path(__file__).resolve().parent / "auto_trader_state" / "account_risk.json"
    if risk_path.exists():
        data = json.loads(risk_path.read_text(encoding="utf-8"))
        for aid in account_ids:
            entry = data.get(aid)
            if entry and entry.get("blocked"):
                print(f"  !! {aid[:8]}... is DRAWDOWN-BLOCKED: {entry.get('reason')} (needs manual re-arm in dashboard)")
        if not any(data.get(aid, {}).get("blocked") for aid in account_ids):
            print("No accounts are drawdown-blocked.")
    else:
        print("No account_risk.json yet (no breaker has ever tripped).")


def _pick_live_data_account(ticker: str, broker_accounts: list, account_asset_classes: dict) -> object | None:
    """Same selection as auto_trader.py's own _pick_live_data_account (kept
    as an independent copy rather than a cross-import between top-level
    scripts, matching this project's existing pattern) - currently only
    AlpacaBroker (equities) exposes get_live_bars. See CLAUDE_NOTES.txt for
    why forex (IG) isn't here yet (its own live-data endpoint has a weekly
    allowance far too scarce for repeated polling)."""
    ticker_asset_class = accounts_module.infer_asset_class(ticker)
    for broker_account in broker_accounts:
        if ticker_asset_class not in account_asset_classes.get(broker_account.account_id, frozenset()):
            continue
        if hasattr(broker_account, "get_live_bars"):
            return broker_account
    return None


def _replay_signal(
    client: PolygonClient, ticker: str, strategy_name: str, params: dict,
    live_data_account: object | None = None,
) -> str:
    if strategy_name not in STRATEGY_REGISTRY:
        return f"unknown strategy '{strategy_name}'"

    from_date = (datetime.now(timezone.utc).date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    to_date = datetime.now(timezone.utc).date().isoformat()
    bars = None
    source = "Polygon"
    if live_data_account is not None:
        try:
            bars = live_data_account.get_live_bars(
                ticker=ticker, from_date=from_date, to_date=to_date, multiplier=1, timespan="minute",
            )
            if bars.empty:
                bars = None
            else:
                source = live_data_account.nickname
        except Exception:  # noqa: BLE001
            bars = None  # fall through to Polygon below

    if bars is None:
        try:
            bars = client.get_aggregates(
                ticker=ticker, from_date=from_date, to_date=to_date, multiplier=1, timespan="minute",
            )
        except Exception as e:  # noqa: BLE001
            msg = f"data fetch failed: {e}"
            if "429" in str(e) or "rate limit" in str(e).lower():
                msg += " (likely just this script and the live bot polling Polygon at the same moment - re-run if it persists)"
            return msg

    if bars.empty or len(bars) < 2:
        return "not enough bars returned"

    strategy = build_strategy(strategy_name, params=params)
    row = bars.iloc[-1]
    current = Bar(
        timestamp=bars.index[-1], open=row["open"], high=row["high"],
        low=row["low"], close=row["close"], volume=row["volume"],
    )
    signal = strategy.on_bar(bars, current)
    if signal.value == "hold":
        return f"HOLD @ {current.close} [{source}]"
    conviction = compute_conviction(strategy, bars, current)
    return f"{signal.value.upper()} @ {current.close} (conviction {conviction:.0%}) [{source}]"


def check_current_signals(control) -> None:
    _hr("LIVE SIGNAL REPLAY (what would fire right now)")
    client = PolygonClient(use_cache=False)

    all_account_ids = list(control.account_ids)
    for group in control.extra_targets:
        for aid in group.get("account_ids", []):
            if aid not in all_account_ids:
                all_account_ids.append(aid)
    linked = accounts_module.list_accounts()
    target_meta = [a for a in linked if a["id"] in all_account_ids]
    broker_accounts = accounts_module.build_broker_accounts(all_account_ids)
    account_asset_classes = {
        a["id"]: frozenset({accounts_module.account_asset_class(a)}) for a in target_meta
    }

    targets: list[tuple[str, str, dict]] = []
    if control.use_roster:
        state = roster.load_roster()
        targets += [
            (e.ticker, e.strategy_name, e.params)
            for e in state.entries if e.status in ("active", "paused")
        ]
    else:
        targets += [
            (t, control.strategy_name, control.manual_strategy_params.get(control.strategy_name, {}))
            for t in control.tickers
        ]
    for group in control.extra_targets:
        for t in group.get("tickers", []):
            targets.append((t, group.get("strategy_name", ""), group.get("strategy_params", {})))

    if not targets:
        print("No active roster entries or extra targets - nothing configured to trade at all.")
        return

    non_hold = 0
    for ticker, strategy_name, params in targets:
        live_data_account = _pick_live_data_account(ticker, broker_accounts, account_asset_classes)
        result = _replay_signal(client, ticker, strategy_name, params, live_data_account)
        print(f"  {ticker:12s} {strategy_name:28s} -> {result}")
        if not result.startswith("HOLD") and "failed" not in result and "unknown" not in result:
            non_hold += 1

    print(f"\n{non_hold}/{len(targets)} target(s) have a non-HOLD signal right now.")
    if non_hold == 0:
        print(
            "  This alone fully explains zero recent trades - every active strategy currently "
            "agrees the market is a HOLD. Not a bug; check back after conditions move."
        )


def check_trade_history() -> None:
    _hr("TRADE HISTORY (execution attempts, last 14 days)")
    if not EXECUTION_LOG_PATH.exists():
        print("No execution_log.jsonl yet - the bot has never attempted an order.")
        return
    cutoff = datetime.now(timezone.utc) - timedelta(days=14)
    counts: dict[str, int] = {}
    fails: dict[str, int] = {}
    with EXECUTION_LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            ts = datetime.fromisoformat(entry["timestamp"])
            if ts < cutoff:
                continue
            day = ts.date().isoformat()
            counts[day] = counts.get(day, 0) + 1
            if not entry.get("success"):
                fails[day] = fails.get(day, 0) + 1

    if not counts:
        print("No execution attempts in the last 14 days.")
        return

    for day in sorted(counts):
        f = fails.get(day, 0)
        suffix = f"  ({f} failed)" if f else ""
        print(f"  {day}: {counts[day]} attempt(s){suffix}")

    today = datetime.now(timezone.utc).date().isoformat()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    if today not in counts and yesterday not in counts:
        print("\n  !! Zero execution attempts today AND yesterday - worth cross-checking the "
              "sections above (process liveness, signals) rather than assuming it's just quiet.")


def main() -> None:
    control = load_control()
    check_process()
    check_event_calendar()

    all_account_ids = list(control.account_ids)
    for group in control.extra_targets:
        for aid in group.get("account_ids", []):
            if aid not in all_account_ids:
                all_account_ids.append(aid)

    check_accounts(all_account_ids)
    check_risk_guards(all_account_ids)
    check_current_signals(control)
    check_trade_history()

    _hr("DONE")
    print("Read top to bottom - each section either clears itself or points at the next one to check.")


if __name__ == "__main__":
    main()
