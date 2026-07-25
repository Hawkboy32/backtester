"""Standalone playlist runner: works through queued scans unattended.

Run it from a terminal (`python scan_runner.py`) or let the dashboard launch
it detached. Either way it coordinates purely through the playlist JSON files
(see backtester.playlist), the same file-based pattern as auto_trader.py — and
for the same reason: a scan can take hours and must survive the dashboard tab
closing.

This process NEVER trades. It only fetches market data and writes backtest
results into scan_history.db, so it has none of auto_trader.py's arming, kill
switch, or live-account machinery. The only control it honours is "stop after
the current ticker".

Exits when the queue has no pending items left, so it can be started and
forgotten.
"""

from __future__ import annotations

import hashlib
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from backtester import playlist
from backtester.data import PolygonClient
from backtester.memory_report import save_report
from backtester.playlist import PlaylistItem, PlaylistStatus
from backtester.scan_db import record_scan
from backtester.scanner import run_scan
from backtester.universe import load_universe

RESULTS_DIR = Path(__file__).resolve().parent / "results"
IDLE_EXIT_MESSAGE = "queue empty — nothing left to run"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _another_instance_alive() -> bool:
    """True if another runner looks alive (fresh heartbeat). Two runners would
    duplicate scans and double the API spend against a 5 req/min limit."""
    status = playlist.load_status()
    if not status.running or not status.last_heartbeat or status.pid == os.getpid():
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(status.last_heartbeat)).total_seconds()
    except ValueError:
        return False
    return age < 300  # a live runner beats at least once per ticker


def run_item(item: PlaylistItem, status: PlaylistStatus) -> None:
    """Run one queued scan to completion, recording it exactly like a manual
    Scanner run so results accumulate in one place."""
    playlist.update_item(item.id, status=playlist.RUNNING, started_at=_now())
    status.current_item_id = item.id
    status.current_label = item.describe()
    status.current_progress = 0.0
    playlist.save_status(status)

    stopped = False

    def on_progress(i: int, total: int, ticker: str) -> None:
        nonlocal stopped
        status.current_ticker = ticker
        status.current_progress = i / total if total else 0.0
        status.last_heartbeat = _now()
        # Cooperative stop: checked per ticker, so a stop request takes effect
        # within one ticker rather than at the end of a multi-hour scan.
        if playlist.load_status().stop_requested:
            stopped = True
            raise KeyboardInterrupt("stop requested")
        playlist.save_status(status)

    # EVERYTHING that can fail for this item lives inside this try — setup
    # (universe lookup, client construction, a bad date string) included, not
    # just run_scan. An early failure escaping here would strand the item in
    # "running" forever and abort the whole queue, which a test caught.
    try:
        universe_df = load_universe(item.universe)
        tickers = universe_df["ticker"].head(item.max_tickers).tolist()
        client = PolygonClient(requests_per_minute=int(item.requests_per_minute))

        # Same checkpoint scheme as the Scanner tab: an interrupted item resumes
        # without re-fetching tickers it already finished.
        config_key = (
            f"{item.universe}|{item.max_tickers}|{sorted(item.strategy_names)}|{item.from_date}|"
            f"{item.to_date}|{item.multiplier}|{item.timespan}|{item.vol_target_enabled}|"
            f"{item.target_vol_ann}|{item.event_filter_enabled}"
        )
        results_dir = RESULTS_DIR / hashlib.sha256(config_key.encode()).hexdigest()[:16]
        checkpoint_path = results_dir / "scan_checkpoint.jsonl"

        results = run_scan(
            tickers=tickers,
            strategy_names=item.strategy_names,
            from_date=item.from_date,
            to_date=item.to_date,
            client=client,
            multiplier=int(item.multiplier),
            timespan=item.timespan,
            starting_cash=item.starting_cash,
            commission_per_trade=item.commission_per_trade,
            slippage_bps=item.slippage_bps,
            max_workers=int(item.max_workers),
            checkpoint_path=checkpoint_path,
            progress_callback=on_progress,
            vol_target_enabled=item.vol_target_enabled,
            target_vol_ann=item.target_vol_ann,
            event_filter_enabled=item.event_filter_enabled,
        )
    except KeyboardInterrupt:
        if stopped:
            # Leave it PENDING, not failed — the checkpoint means resuming is cheap.
            playlist.update_item(item.id, status=playlist.PENDING, started_at=None)
        raise
    except Exception as e:  # noqa: BLE001
        playlist.update_item(
            item.id, status=playlist.FAILED, finished_at=_now(), error=f"{type(e).__name__}: {e}"
        )
        status.last_error = f"{item.describe()}: {e}"
        playlist.save_status(status)
        return

    meta = {
        "run_at": _now(),
        "universe": item.universe,
        "num_tickers": len(tickers),
        "from_date": item.from_date,
        "to_date": item.to_date,
        "multiplier": int(item.multiplier),
        "timespan": item.timespan,
        "strategy_names": item.strategy_names,
        "vol_target_enabled": item.vol_target_enabled,
        "target_vol_ann": item.target_vol_ann,
        "event_filter_enabled": item.event_filter_enabled,
        "source": "playlist",
    }
    try:
        run_id = record_scan(meta, results)
    except Exception as e:  # noqa: BLE001
        # The scan itself succeeded but couldn't be persisted — say so plainly
        # rather than marking the item done with nothing to show for it.
        playlist.update_item(
            item.id, status=playlist.FAILED, finished_at=_now(),
            error=f"scan finished but recording it failed: {type(e).__name__}: {e}",
        )
        status.last_error = f"{item.describe()}: could not record results: {e}"
        playlist.save_status(status)
        return

    try:
        save_report(results, meta, results_dir)
    except Exception:  # noqa: BLE001
        pass  # the standalone txt/json export is a nicety; the DB row is what matters

    playlist.update_item(
        item.id, status=playlist.DONE, finished_at=_now(), run_id=run_id, num_results=len(results)
    )
    status.items_done += 1
    status.current_ticker = None
    status.current_progress = 1.0
    playlist.save_status(status)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    if _another_instance_alive():
        print("Another scan runner appears to be running (fresh heartbeat) — exiting.")
        return

    status = playlist.load_status()
    status.running = True
    status.pid = os.getpid()
    status.last_heartbeat = _now()
    status.last_error = None
    status.stop_requested = False  # a fresh start clears any stale stop request
    status.items_done = 0
    playlist.save_status(status)

    print(f"Scan runner starting (pid={os.getpid()}).")
    try:
        while True:
            items = playlist.load_playlist()
            status.items_total = len(items)
            pending = playlist.next_pending(items)
            if pending is None:
                print(IDLE_EXIT_MESSAGE)
                break
            print(f"Running: {pending.describe()}")
            try:
                run_item(pending, status)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # noqa: BLE001
                # Belt-and-braces: run_item already marks its own failures, but
                # one unexpected item must never end the whole queue. Mark it
                # failed (so next_pending moves on) and keep going.
                playlist.update_item(
                    pending.id, status=playlist.FAILED, finished_at=_now(),
                    error=f"unhandled: {type(e).__name__}: {e}",
                )
                status.last_error = f"{pending.describe()}: {e}"
                playlist.save_status(status)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped (stop requested or interrupted).")
    except Exception:  # noqa: BLE001
        status.last_error = traceback.format_exc(limit=3)
        print(status.last_error)
    finally:
        status.running = False
        status.current_item_id = None
        status.current_label = None
        status.current_ticker = None
        status.stop_requested = False
        status.last_heartbeat = _now()
        playlist.save_status(status)
        print("Scan runner stopped.")


if __name__ == "__main__":
    main()
