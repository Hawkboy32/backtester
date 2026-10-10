"""Activity node: what Chopper actually did today. Read-only - no broker
call, no mutation of any state file. Three tiers of real data, same ones
identified by exploring the live state before this was built:
`auto_trader_state.json` for today's trade count and the single latest
signal, `current_signals.json` for the latest signal PER TICKER, and
`live_trades.db` for closed round-trips with real P&L.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtester import auto_trader_state, current_signals, live_trades  # noqa: E402


def gather_activity(*, recent_trades_limit: int = 20) -> dict:
    """Returns a structured summary - no prose yet, that's the writer
    node's job. Never raises: every sub-read already tolerates a missing/
    unreadable state file on its own (see auto_trader_state.load_status's
    own fallback), so this just passes that through."""
    status = auto_trader_state.load_status()
    signals = current_signals.load_signals()
    recent_trades = live_trades.list_recent_trades(limit=recent_trades_limit)

    todays_closed_trades = [t for t in recent_trades if (t.get("exit_time") or "").startswith(_today())]

    return {
        "running": status.running,
        "trades_today": status.trades_today,
        "last_signal": status.last_signal,
        "last_error": status.last_error,
        "per_ticker_signals": [
            {"ticker": ticker, "signal": info.get("signal"), "strategy_name": info.get("strategy_name"), "conviction": info.get("conviction")}
            for ticker, info in signals.items()
        ],
        "closed_trades_today": [
            {
                "ticker": t.get("ticker"),
                "strategy_name": t.get("strategy_name"),
                "pnl": t.get("pnl"),
                "pnl_pct": t.get("pnl_pct"),
                "account_id": t.get("account_id"),
            }
            for t in todays_closed_trades
        ],
    }


def _today() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
