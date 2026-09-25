# Backtester

A continuously-running trading system integrating six real brokerage APIs
(Alpaca, Interactive Brokers, Coinbase, Kraken, OANDA, IG) across equities,
FX, and crypto, with an adaptive strategy-selection framework — nothing
reaches live trading without first clearing a rigorous backtest gate.

Built and maintained solo via an AI-pair-programming workflow (Claude Code).

**Full write-up, real screenshots, and the story behind specific bugs found
and fixed:** [hawkboy32.github.io/backtester.html](https://hawkboy32.github.io/backtester.html)

## What's in here

- `src/backtester/` — core engine: strategies, broker integrations,
  execution/sizing, roster management, risk controls
- `app.py` — the Streamlit dashboard ("Holotable")
- `auto_trader.py` — the standalone live-trading loop
- `copy_trade_executor.py` — a copy-trading signal engine built on top of
  the same execution path, reverse-engineered against an undocumented
  third-party API
- `test_*.py` — regression tests, one per real incident found and fixed
  (see each test's own docstring for what actually happened)

## License

All rights reserved — see [LICENSE](LICENSE). This repository is public for
portfolio and evaluation purposes; it is not licensed for reuse.
