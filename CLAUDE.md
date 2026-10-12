# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

"Chopper" — a continuously-running, **live-money** multi-broker algorithmic trading system. `src/backtester/` is the installed Python package (backtests, strategies, broker integrations, risk controls, state). Three separate top-level entry points run independently of each other, coordinating only through shared state files, not in-process calls:

- **`app.py`** — the Streamlit dashboard ("Holotable"). Lets a human configure accounts, strategies, roster, risk settings, and arm/disarm/kill the live trader. Does **not** itself place trades.
- **`auto_trader.py`** — the actual live-trading loop. A standalone process (launched from a terminal, via `pythonw.exe` on login, or as a detached subprocess from the dashboard) that polls, computes signals, sizes orders, and submits them to real brokers.
- **`ibgateway_watchdog.py`** — watches IB Gateway's own reachability and notifies (Gateway force-logs-out ~daily; this catches cases IBC's auto-relogin can't close, e.g. a stuck 2FA prompt). Since 2026-10-12 it also **starts Gateway when its process is gone and restarts it when it is alive but not logged in** — every attempt is a login and therefore a 2FA push to the user's phone, so it is hard-limited (one an hour, three in 24 hours, none 23:00–07:00, none while a paper Gateway or TWS holds the login). Create `ibgateway_watchdog.pause` next to it to switch that off — do this before stopping Gateway on purpose (an upgrade, say), or it will be started again ten minutes later. Not the trading loop.
- **`auto_trader_watchdog.py`** — the script that used to relaunch `auto_trader.py` if it died. **Retired on this machine 2026-10-11**: Sentrywatch (`../sentrywatch`) now starts and supervises `auto_trader.py`, `ibgateway_watchdog.py` and the mobile backend at login. The file is kept (and `../ChopperCommercial` still ships its copy), but nothing here launches it any more. It also has a known pid-reuse bug, written up in `CLAUDE_NOTES.txt` 2026-10-10 — read that before ever putting it back into service.
- **`daily_briefing/`** — a newer, separate LangGraph multi-agent add-on with its **own `.venv`** (isolated from the main project's dependencies). Strictly read-only — gathers activity/risk data and writes a plain-English summary; never places, modifies, or cancels an order.

## Common commands

```bash
# Dashboard
.venv\Scripts\streamlit.exe run app.py        # or: run_dashboard.bat
python scripts\setup_auth.py                  # first-time login setup (creates auth_config.yaml)

# Live trader (real orders once armed from the dashboard)
.venv\Scripts\pythonw.exe auto_trader.py      # pythonw so a crash has nowhere silent to go except logging_setup's log file
# or, for console output while developing:
.venv\Scripts\python.exe auto_trader.py

# Supervision: Sentrywatch starts auto_trader.py and ibgateway_watchdog.py at login and
# relaunches them. Don't start either by hand while it is running - the trader's own guard
# and the gateway watchdog's will refuse a second copy, but there is no reason to test them.
# Sentrywatch's dashboard: http://127.0.0.1:8972 (or double-click ..\sentrywatch\installed\sentrywatchw.exe)
.venv\Scripts\python.exe test_ibgateway_watchdog_singleton.py   # the gateway watchdog's guard, against real processes
.venv\Scripts\python.exe test_ibgateway_watchdog_recovery.py    # when it may start/restart Gateway; never touches the real one

# Tests — root-level: standalone scripts, NOT pytest. Each is one regression
# test for one real incident (see its own docstring). Run individually:
.venv\Scripts\python.exe test_roster_reevaluation_single_active.py
.venv\Scripts\python.exe test_buying_power_buffer.py
# (same pattern for the other test_*.py / verify_*.py files in the repo root)

# daily_briefing — its OWN venv, separate dependency set (LangGraph, anthropic SDK)
cd daily_briefing
.venv\Scripts\python.exe run_once.py              # one real run, prints everything, sends nothing
.venv\Scripts\python.exe run_once.py --notify     # same, but actually sends the notification
.venv\Scripts\python.exe scheduler.py             # runs continuously, once/day after a cutoff hour
.venv\Scripts\pytest.exe                          # daily_briefing DOES use real pytest (tests/, conftest.py)
```

There is no pytest suite for the main package — root-level `test_*.py`/`verify_*.py` files are self-contained scripts with their own `if __name__ == "__main__"` blocks, run directly.

## High-level architecture

**Broker abstraction.** `src/backtester/brokers/base.py` defines `BrokerAccount` (ABC): `get_account_snapshot()`, `get_positions()`, `get_equity_history()`, `submit_market_order()` (abstract — every broker must implement these), plus non-abstract hooks with safe defaults (`get_fees()` raises `NotImplementedError` where a broker has no uniform fee endpoint, `get_market_clock()` returns `None` for 24/7 venues, `get_open_order_tickers()` defaults to an empty set). Also defines `OrderSide`, `AccountSnapshot`, `Position`, `OrderResult`, `BrokerFee` (with `.is_funding` to split trading costs from currency-conversion costs). Each broker in `brokers/` (`alpaca.py`, `ibkr.py`, `coinbase.py`, `kraken.py`, `oanda.py`, `ig.py`, `etoro.py`, `tastytrade_broker.py`) implements this interface, so `execution.py` and `app.py` never need to know which broker they're talking to. `supports_fractional_shares` and `native_reference_price()` are per-broker overrides worth knowing about (IBKR rejects fractional equity orders; Coinbase overrides `native_reference_price` to avoid mis-sizing when a signal's price feed currency differs from the account's execution currency).

**`accounts.py` — the central broker-account registry.** `BROKER_META` is the per-broker-type config table (credential field labels, paper-mode support, market hours, supported asset classes, whether it's a "gateway" broker like IBKR with no API key/secret). Credentials are encrypted via `keyring` (OS credential store); `broker_accounts.json` (gitignored) holds only non-secret metadata (nickname, broker, paper/live flag, last-4 hint, and for IBKR non-secret `conn_params`). `build_broker_accounts()` instantiates live `BrokerAccount` objects from stored accounts — this is the one place that knows how to construct every broker type. `infer_asset_class()` / `account_asset_class()` decide which target accounts a given ticker/strategy combo is even eligible for (so a forex-only IG account can sit permanently in `control.account_ids` without erroring on equity cycles).

**State files (`auto_trader_state/`, gitignored).** `src/backtester/auto_trader_state.py` defines and reads/writes:
- `control.json` → `AutoTraderControl` — what the dashboard tells `auto_trader.py` to do (enabled/killed/allow_live, account_ids, tickers, strategy, sizing mode/value, roster mode toggle, risk-guard toggles, protective-exit settings, `extra_targets` for concurrent secondary target groups, per-account sizing/position-mode overrides).
- `status.json` → `AutoTraderStatus` — what `auto_trader.py` is actually doing (running/pid/last_heartbeat, trades_today, last_signal/last_error, roster-check dates).
- `atomic_write_text()` — writes via temp-file + `os.replace` so a concurrent reader never sees a half-written file (a plain truncate-then-write used to get read as "corrupt" and kill a healthy bot).
- `read_state_json()` / `from_dict()` — deliberately distinguish "file missing" (safe default) from "file present but unparseable" (raises `StateFileUnreadable`, refuses to overwrite) and tolerate **new** fields an **older** running process doesn't know about. Both exist because of two real live data-loss incidents (documented in the code: a corrupted roster write destroyed 3,229 records). **Read the docstrings in this file before changing its schema shapes** — they explain exactly which failure mode each safeguard prevents.
- Fail-closed philosophy throughout: a missing/corrupt control file means "treat as killed", never "keep trading with stale/default settings."

**Singleton heartbeat-guard pattern.** `auto_trader.py`'s `_another_instance_alive()` requires BOTH a live OS PID *and* a recently-updated heartbeat before refusing to start a second instance — checking only one of the two was each independently proven unreliable live (a hard-killed process's heartbeat can look fresh for minutes; a hung-but-not-crashed process still holds a live PID). The same two-signal shape is reused twice more: `auto_trader_watchdog.py` uses it to decide whether `auto_trader.py` is healthy, and `ibgateway_watchdog.py` uses it as its own singleton guard (added 2026-10-10, state in the gitignored `ibgateway_watchdog_state.json`) — before that a second manual launch simply started a second watchdog. `ibgateway_watchdog.py` has no `--help` or dry-run path: running it at all starts the real loop.

**Roster / strategy selection.** `src/backtester/roster.py` is the adaptive promotion/demotion engine: `evaluate_roster()` promotes (ticker, strategy) combos based on backtest score (via `ranking.rank_combos`) but demotes/pauses them based on **real** live/paper trade outcomes (via `live_trades.recent_performance`) — a combo with a great backtest can still get paused for actually losing money right now. State persists to `auto_trader_state/roster.json` (+ an append-only `roster_events.jsonl`). Promotion only happens via a deliberate "re-evaluate roster" dashboard action (scanning is expensive); demotion checks run every `auto_trader.py` cycle. The `score_fn` seam is explicitly left open for a future ML meta-model.

**Risk guards.**
- `account_risk.py` — account-level max-drawdown circuit breaker. Hard-blocks new entries once equity falls more than `max_drawdown_pct` below that account's peak; does **not** self-heal — requires a manual re-arm from the dashboard, which also resets the peak.
- `daily_pnl_guard.py` — a lighter, same-day "giveback" guard: blocks new entries once today's profit has retraced more than `giveback_pct` off *today's own* intraday peak. Resets automatically the next day, no manual re-arm. Independent of, and skipped if redundant with, the breaker above.
- `events.py` — known risk-event day gating (e.g. FOMC), on by default (`block_event_days`), unlike the other opt-in guards.

**Live state/record-keeping beyond control/status:**
- `src/backtester/live_trades.py` — SQLite (`live_trades.db`, gitignored) record of **realized** round-trip trades from live/paper trading, attributed by strategy. This is the ground truth the roster's demotion rules read from (distinct from `execution_log.py`'s raw unattributed order-submission log).
- `src/backtester/current_signals.py` — JSON snapshot (`auto_trader_state/current_signals.json`) of the most recently computed signal per ticker, written by `auto_trader.py` so a second process (the mobile backend, `daily_briefing/`) can read "what would the strategy say right now" without its own Polygon fetch/rate-limit budget.
- `src/backtester/notifications.py` — push notifications (ntfy by default) for trade/risk events. `notify()` deliberately never raises — a notification failure must never block, delay, or abort a trade.

## File / directory map

**Top-level scripts**
| Path | What it is / when you'd touch it |
|---|---|
| `app.py` | Streamlit dashboard. Touch for UI/config changes, account linking flows, chart/report additions. Does not place trades itself. |
| `auto_trader.py` | The live-trading loop. Touch with real caution — polling cadence, signal→order flow, roster/manual mode dispatch, all risk-guard wiring, singleton guard. |
| `auto_trader_watchdog.py` | Relaunches `auto_trader.py` if it dies mid-session. Touch for relaunch/health-check logic only. |
| `ibgateway_watchdog.py` | Watches IB Gateway reachability, notifies on prolonged outage, and starts/restarts Gateway within hard limits (`_recovery_action()` is the whole decision; its limits are constants at the top). Runs `../ibc/IBC/StartGateway.bat`. Only relevant with an IBKR account linked. |
| `copy_trade_executor.py` / `copy_trader.py` | Copy-trading signal engine built on the same execution path, reverse-engineered against an undocumented third-party (eToro) API. |
| `challenge_trader.py` / `challenge_notifier.py` | Kraken Funded prop-challenge ladder logic (see `src/backtester/challenge_state.py`). |
| `restart_all.py`, `wait_then_restart.py` | Process-restart helpers for the live stack. |
| `reconcile_fills.py`, `diagnose_bot.py`, `diagnose_turn_before_loss.py` | Ad hoc/pre-flight diagnostic scripts — not part of the running system. |
| `scripts/setup_auth.py` | First-time dashboard login setup. |
| `scripts/refresh_sp500_list.py`, `refresh_nasdaq100_list.py` | Universe refresh helpers. |
| `test_*.py`, `verify_*.py` (repo root) | Standalone regression scripts, one per real live incident — read each one's own docstring for what actually happened. |
| `daily_briefing/` | LangGraph multi-agent read-only briefing add-on; own `.venv`. See its own `README.md`. Entry points: `run_once.py`, `scheduler.py`; graph in `graph.py`; nodes in `activity_node.py`/`risk_node.py`/`writer_node.py`; tests in `tests/` (real pytest). |

**`src/backtester/` (load-bearing modules)**
| Module | What it is / when you'd touch it |
|---|---|
| `brokers/base.py` | `BrokerAccount` ABC + shared dataclasses. Touch when adding a capability every broker needs to expose. |
| `brokers/*.py` (alpaca, ibkr, coinbase, kraken, oanda, ig, etoro, tastytrade_broker) | One file per broker integration. Touch the relevant one for broker-specific execution/auth/quirk fixes. |
| `accounts.py` | Central broker-account registry: `BROKER_META`, credential storage (keyring), `build_broker_accounts()`. Touch when adding a new broker type or changing how accounts are linked/built. |
| `auto_trader_state.py` | `control.json`/`status.json` schema + atomic read/write helpers. Touch with care — schema changes must stay backward-compatible with `from_dict`'s unknown-key tolerance; see its docstrings for the two real data-loss incidents this file exists to prevent. |
| `roster.py` | Adaptive promotion/demotion engine. Touch for roster-selection logic, caps (top-N/per-strategy/per-sector), pause/promote rules. |
| `ranking.py` | Backtest-score ranking feeding roster promotion. |
| `account_risk.py` | Account-level max-drawdown circuit breaker. |
| `daily_pnl_guard.py` | Same-day P&L giveback guard. |
| `execution.py` | Order sizing (fixed-dollar / %-equity / sliding-%, vol-target) and cross-account order execution. |
| `live_trades.py` | SQLite realized-trade history (`live_trades.db`) — ground truth for roster demotion. |
| `execution_log.py` | Raw order-submission log (no attribution/round-trip concept — distinct from `live_trades.py`). |
| `current_signals.py` | Shared latest-signal-per-ticker snapshot, read by the mobile backend/`daily_briefing`. |
| `notifications.py` | ntfy-based push notifications for trade/risk events; never raises. |
| `position_attribution.py` | Attributes open positions to the strategy/combo that opened them. |
| `challenge_state.py` | Kraken Funded prop-challenge ladder state. |
| `copy_roster.py` / `copy_trading.py` | Copy-trading roster/signal logic backing `copy_trade_executor.py`. |
| `engine.py` | The core backtest engine (`BacktestEngine`). |
| `strategies/` | One file per trading strategy (indicator-based signal generators), plus `indicators.py` and `strategy_regime` helpers. |
| `scanner.py` / `universe.py` | Ticker universe + scan-result generation feeding ranking/roster. |
| `data.py` | `PolygonClient` — market-data fetch/cache layer. |
| `events.py` | Known risk-event day calendar (FOMC etc.) for `block_event_days`. |
| `deposits.py`, `tax.py` | Deposit/conversion tracking and GBP tax calc. |
| `two_factor.py`, `keystore.py` | Dashboard auth/2FA support. |
| `source_stamp.py` | Stamps which code revision a running process loaded, to flag a stale process in the dashboard. |
| `heartbeat.py`, `logging_setup.py` | Process heartbeat + log-file setup (critical for `pythonw.exe`, which has no console to print to). |
| `risk_presets.py` | Named bundles of risk-guard settings ("Conservative"/"Moderate"/"Aggressive") applied as a group from the dashboard. |
| `metrics.py`, `volatility.py`, `conviction.py` | Scoring/regime-classification/vol-targeting helpers used by ranking, roster, and sizing. |

## Known gotchas

- **This is LIVE, with REAL MONEY.** Any change touching `auto_trader.py`, anything in `brokers/`, `execution.py`, or order placement/sizing needs real caution — prefer the conservative read of ambiguous requirements, add a regression test for anything that was a real incident, and don't change sizing/risk-guard defaults without being asked.
- **User's stated risk posture: family wealth-building, not get-rich-quick — always default to the cautious option** when a choice isn't specified.
- **IBKR Live and Paper gateways can't be logged in concurrently** — a "connection refused" on one is often just the other one holding the session, not a real outage.
- **IB Gateway needs IBC (IB Controller) + `ibgateway_watchdog.py`** for its forced ~daily restart/relogin; the watchdog's grace period exists because that restart cycle is expected to have a short gap even when healthy. IBC only restarts a Gateway that is running and logged in; a dead or logged-out one is the watchdog's job (see above).
- **Gitignored state/secret files — never hand-edit:** `auto_trader_state/control.json` and `status.json` (live coordination state), `broker_accounts.json` (account metadata — real credentials live in the OS keyring, not this file), `.env` (API keys/secrets), `live_trades.db` (realized-trade ledger). Go through the dashboard or the module's own read/write helpers, never a raw text edit.
- Root-level `test_*.py`/`verify_*.py` scripts are regression tests for **real** past incidents, not illustrative examples — if one starts failing, treat it as a live bug resurfacing, not a stale test to delete.
- `daily_briefing/` has a **separate `.venv`** and its own dependency set (LangGraph, Anthropic SDK) — don't assume the main project's `.venv` has what it needs, or vice versa.
- **Other things launch and read this repo by path.** Sentrywatch's config (`%APPDATA%\Sentrywatch\sentrywatch.toml`) launches `auto_trader.py` and `ibgateway_watchdog.py` from this folder's `.venv` and reads `auto_trader_state/status.json` and `ibgateway_watchdog_state.json` as their heartbeats — renaming either script, either state file, or the `last_heartbeat` key inside them silently breaks supervision. A Startup-folder shortcut starts the dashboard; `../Mobile_App/backend` imports this package as an editable install and reads `auth_config.yaml` and `.env` from here. Renaming a top-level script or moving the folder breaks those — the full list is in `../CLAUDE_NOTES.txt`.
- **`../ChopperCommercial/src/backtester` is a separate copy of this package**, not a link. A fix here does not reach it.

## Pointers

- **`../CLAUDE_NOTES.txt`** — the workspace master index: every sibling project, cross-repo dependencies, and what runs at login.
- **`CLAUDE_NOTES.txt`** (50,000+ words) — a dated, chronological running history/incident log for this project. Don't read it wholesale; grep it for a date, bug name, or feature when you need the backstory on something referenced above (e.g. "2026-08-24 outage", "roster-gap bug").
- **`README.md`** — short public-facing project overview and link to a full write-up.
- **`daily_briefing/README.md`** — architecture and running instructions for that sub-project (graph shape, nodes, verified-live notes).
- **`BROKER_EXPANSION_RESEARCH.md`**, **`ROADMAP_AND_DASHBOARD_PLAN.md`** — planning docs, not source of truth for current behavior; check the code first.
