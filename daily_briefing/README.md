# daily_briefing

A multi-agent daily briefing for Chopper, orchestrated with **LangGraph**
instead of a hand-rolled script — two real data-gathering agents run in
parallel, fan into a writer agent that turns both into a plain-English
summary. Strictly read-only: nothing here places, modifies, or cancels an
order, same discipline as every other AI piece built this session
(Sentrywatch's copilot, the MCP server, docs-assistant).

## The graph

```
START ──▶ activity ──┐
                      ├──▶ writer ──▶ END
START ──▶ risk ───────┘
```

- **`activity_node.py`** — today's trading activity: `trades_today` and
  the latest signal from `auto_trader_state.json`, every ticker's latest
  signal from `current_signals.json`, today's closed round-trips with
  real P&L from `live_trades.db`.
- **`risk_node.py`** — the one node that touches the outside world: real
  authenticated broker connections (`accounts.build_broker_accounts()`),
  real equity/cash/positions per account, cross-referenced against the
  drawdown and daily-giveback circuit breakers. One broker failing is
  isolated to that account's own `error` field — it doesn't take the rest
  of the run down.
- **`writer_node.py`** — same dual-provider shape as every other AI piece
  this session: Ollama by default (free, verified live), Claude via
  `provider="claude"` (real Anthropic SDK, **verified live 2026-10-09**,
  ~$0.015/call at `claude-sonnet-5-5` pricing).

## Running it

```
.venv\Scripts\pip install -r ..\requirements.txt
.venv\Scripts\pip install langgraph anthropic
.venv\Scripts\python run_once.py                 # one real run, prints everything, sends nothing
.venv\Scripts\python run_once.py --notify         # same, but actually sends the notification
.venv\Scripts\python scheduler.py                 # runs continuously, once/day after the cutoff hour
```

Its own venv, separate from the live trading bot's — this adds LangGraph's
dependency footprint on top of (not instead of) Chopper's own real
dependencies, deliberately isolated from whatever `auto_trader.py` itself
runs on.

## Verified live, 2026-10-09

Ran `run_once.py` for real against every currently linked broker account
(10 accounts: Kraken, two IBKR, IG, OANDA, two Alpaca, Coinbase, two
eToro). Two real, pre-existing conditions surfaced correctly rather than
crashing the run: IBKR Paper's connection was refused (a real, already-
known IB Gateway live/paper exclusivity issue), and ETLive returned a real
404 — both landed as that one account's own `error` field, exactly the
isolation this was built to prove, while the other eight accounts' real
data (including KRKAPI's real tripped drawdown breaker) came through
clean. The writer's summary correctly mentioned both failures and the
breaker by name, inventing nothing not present in the data.

## Claude path verified live, 2026-10-09

Gathered fresh real activity/risk data (free - local files + the same
broker reads as above), then called `write_briefing(..., provider="claude")`
for the one real paid call: 4,897 input / 508 output tokens, ~$0.015.
The result was not just stylistically different from Ollama's - it caught
something Ollama's briefing didn't: KRKAPI's drawdown-breaker reason text
("equity $13.77 was more than 15% below peak $16.22") didn't match the
account's actual current equity ($5.86), and Claude flagged the mismatch
as worth checking instead of repeating the stale number silently. A real,
specific instance of Claude noticing something in the data Ollama didn't
comment on - not a generic "Claude writes better prose" claim.

## Not yet done

- **`scheduler.py` has not been left running continuously** — every real
  run above was a manual, reviewed `run_once.py` call, not `--notify`.
  Turning the scheduler on (so it actually messages the phone every day)
  is a separate decision, not implied by having built and verified it.
