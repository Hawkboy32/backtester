# Broker expansion research — forex / CFDs / multi-market longevity

_Researched 2026-07-21. Goal (user): explore expanding beyond Alpaca's US stocks/ETFs —
access to more markets (forex, gold, indices), CFD trading, and a broker with longevity.
This is a reference for a FUTURE decision, not a commitment to build. Verify API details
against the live docs again before building — broker APIs change._

## The bottom line up front

- **IG, Capital.com, FOREX.com** all have genuine, documented, public REST APIs with demo
  modes — technically integratable to the same standard as the existing brokers. BUT they
  are fundamentally **CFD / leveraged-derivative** shops. That is a real risk-category
  change from Alpaca's unleveraged share dealing, and sits in tension with the project's
  stated family-wealth / "don't blow through funds" ethos.
- **Interactive Brokers (IBKR)** is the odd one out in a good way: it trades **real**
  assets (shares, futures, spot forex, options, bonds) globally, can run unleveraged, is
  professional-grade, and has proper paper trading. Its cost is integration complexity (a
  local Gateway/TWS process must run alongside the bot). It's the natural "graduate from
  Alpaca to a serious multi-asset broker" step for longevity.
- **Two blockers sit UNDER the API question and matter more than any of them:**
  1. **Leverage risk** — CFDs are geared; brokers must legally display that ~70–80% of
     retail CFD accounts lose money. An automated bot on leveraged CFDs is close to the
     opposite of the project's risk posture unless leverage is pinned to 1x and handled
     with real care. This is a decision to make deliberately, not a side effect of adding
     a broker.
  2. **No forex/CFD backtest data** — the entire data/engine path is Polygon **equities**.
     None of these instruments plug into it. Without a forex/CFD historical data source we
     cannot do the "backtest first, then decide" step (the exact discipline that just saved
     us from blindly paper-trading gold). And the gold-ETF test (2026-07-21) showed the
     strategies don't even transfer from stocks to a gold ETF — forex/CFDs are further
     still, so there's real doubt the current strategies work on them at all. Adding
     execution without matching data = trading these markets blind.

## Per-broker assessment

### IG (IG Group) — CFDs / spread betting / forex
- **API:** Documented REST + streaming (Lightstreamer). Guide: https://labs.ig.com/rest-trading-api-guide.html
- **Base URLs:** demo `https://demo-api.ig.com/gateway/deal`, live `https://api.ig.com/gateway/deal`
- **Auth:** API key + account credentials → session tokens.
- **Demo:** Yes — full demo account API. Good for safe testing.
- **Instruments:** CFDs, spread betting (UK, tax-specific, gambling-regulated), forex,
  limited shares. API is primarily the CFD/spread-bet dealing API.
- **Python:** community lib `trading-ig` (mature) exists.
- **Verdict:** Cleanest CFD API of the three; UK-established. Leveraged-product caveat applies.

### Capital.com — CFDs only
- **API:** Documented REST. Docs: https://open-api.capital.com/
- **Auth:** Generate API key in Settings (requires 2FA on). `POST /session` with
  `X-CAP-API-KEY` → returns `CST` + `X-SECURITY-TOKEN` headers used on every request.
  **Sessions expire after 10 minutes** (needs re-auth handling).
- **Demo:** Yes — `https://demo-api-capital.backend-capital.com/`.
- **Instruments:** Everything is a **CFD** (forex, shares, commodities incl. gold, indices,
  crypto — all as CFDs). No real-asset ownership.
- **Python:** community libs (`capitalcom-python`) exist.
- **Verdict:** Easy, modern API, broad CFD market access, demo. Purely leveraged CFDs.

### FOREX.com (StoneX / GAIN Capital) — forex + CFDs
- **API:** REST API for automation. Page: https://www.forex.com/en/trading-tools/api-trading/
- **Access:** **Must contact their support to request API access** and receive an app key
  (extra gate vs the others). NOTE: web searches also surface FXCM's readthedocs — that is
  a DIFFERENT broker (FXCM), do not conflate with FOREX.com.
- **Demo:** Yes.
- **Instruments:** Forex + CFDs. Leveraged.
- **Verdict:** Capable, but the manual access-request step and thinner public docs make it
  the highest-friction of the three CFD options.

### Interactive Brokers (IBKR) — REAL multi-asset, global
- **API:** Two options —
  - **Client Portal Web API** (REST): https://www.interactivebrokers.com/campus/ibkr-api-page/ibkr-api-home/
    Two-tier session (read-only portal + brokerage session). IBKR **Pro** account required.
    Test with paper first. Still needs a local gateway session + periodic re-auth.
  - **TWS API** (asynchronous): full-featured, stable, low latency — but requires Trader
    Workstation or IB Gateway running locally (heavy dependency, exactly why the project
    parked IBKR as "revisit later").
  - Constraint: one brokerage session per username at a time.
- **Demo:** Yes — proper paper trading.
- **Instruments:** **Real** stocks, options, futures, spot forex, bonds — globally.
  Unleveraged-capable. Gold via futures (GC) or ETFs.
- **Verdict:** Most capable and the best fit for the project's "own real assets, no forced
  leverage, longevity" values. Heaviest integration (gateway process). Best long-term bet.

## What each would need to fit the existing architecture
Every broker must implement the `BrokerAccount` ABC (`brokers/base.py`):
`get_account_snapshot`, `get_positions`, `get_equity_history`, `submit_market_order`,
`get_market_clock`, `get_open_order_tickers`, plus `account_id`/`nickname`/`is_paper`.
That part is well-trodden (4 brokers already done). The genuinely NEW work is NOT the
execution adapter — it's:
- **A forex/CFD historical data source** for the backtest/scanner/GARCH path (Polygon has
  forex + crypto feeds; or the broker's own history endpoints; or another vendor). Without
  this, no "backtest first" for these instruments.
- **Instrument/symbol model** — CFDs and forex pairs aren't equity tickers; universe,
  sizing (contracts/lots vs shares), and the flat-long-only engine assumptions need review.
- **Leverage handling** — explicit, capped, defaulting to 1x; the account-risk circuit
  breaker (`account_risk.py`) becomes even more important with geared products.

## Proposed sequence (cautious, longevity-first)
1. **Finish the current equities paper validation** — prove the core system + strategies on
   what already works before widening scope.
2. **Decide the leverage question** deliberately — are leveraged CFDs acceptable given the
   family-wealth goal, and if so at what capped leverage? This gates the whole CFD path.
3. **Solve forex/CFD historical data FIRST** (a data source), so any new instrument can go
   through the same "backtest first, then decide" gate gold just went through.
4. **Then pick a broker to match intent:**
   - Longevity / real assets → **Interactive Brokers** (accept the gateway complexity).
   - Leveraged FX/CFD markets specifically → **IG** or **Capital.com** (cleanest APIs),
     only after steps 2–3.
5. **Validate strategies on the new instrument class** before any live/paper auto-trading —
   the gold test already showed equity-tuned strategies may not transfer.

## Open questions for the user
- Willing to trade leveraged CFDs at all? If yes, what max leverage (1x recommended)?
- Priority: breadth of markets now (CFD broker) vs a solid long-term real-asset broker
  (IBKR) even if it's more setup?
- Appetite for sourcing forex/CFD historical data (needed to validate before trading)?
