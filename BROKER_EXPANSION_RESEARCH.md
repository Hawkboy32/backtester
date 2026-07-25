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

---

# 2026-07-25 update — working the two blockers

_Since the above was written, IBKR (#30) has been BUILT and live-tested against a real paper
account. That changes the picture materially, and is the single most important thing on this
page now. Research only — nothing here is built, and no leverage decision has been made._

## The reframe: you may not need IG at all

The original goal was "access more markets — forex, gold, indices". IG was the candidate
because it has the cleanest CFD API. But **IBKR is now connected and working**, and IBKR
trades **real spot forex** — unleveraged-capable, actual asset ownership, through a gateway
connection that is already proven (paper account [redacted-demo-account-id], snapshot/positions/clock/orders
all verified 2026-07-25).

So the honest question is no longer "how do we integrate IG?" but:

> **Do you actually want leveraged CFDs, or did you want access to forex?**

Because those have different answers:

| If what you want is... | The right move is... |
|---|---|
| Forex exposure, owning the position, no gearing | **Extend the existing IBKR adapter to spot FX.** No new broker, no new credentials, no leverage decision, reuses a working connection. |
| Spread betting specifically (UK tax treatment) | IG — it is genuinely the only one of these that offers it. This is a tax question, not a technical one; worth an accountant's view before it drives an architecture decision. |
| Geared exposure — controlling more than you put in | IG/Capital.com CFDs — and this is the risk-category change the whole document warns about. |

**Recommendation: extend IBKR to spot forex rather than integrating IG**, unless spread
betting's tax treatment is the actual draw. It is less work, less new surface, no new
credential store, and it does not require accepting leverage. IG stays documented here if
that changes.

## Blocker (a): the leverage decision — laid out for a deliberate call

This has been "pending" since 21 July. It cannot be resolved by research, only by choosing.
The facts, stated plainly:

- CFD brokers are legally required to display that **~70–80% of retail CFD accounts lose
  money**. That figure is for humans trading discretionarily; an automated system is not
  obviously better and may be worse, because it can act on a broken edge faster than you
  can notice.
- Leverage does not improve an edge. It multiplies whatever is already there — including a
  negative one. This system currently has **no demonstrated edge**: one live trade in week
  one, and `live_trades.db` still holds zero completed round trips.
- The engine is long-only, flat-entry, no shorting, and cannot currently express a margin
  position. Gearing it is not a config flag; it is new position-sizing, new margin
  accounting, and a real risk of a margin call the existing `account_risk.py` breaker was
  never designed for.
- Against the stated goal — building family wealth, defaulting to the cautious option —
  **the recommendation is to decline leverage entirely, or cap it hard at 1x** (i.e. use a
  CFD purely as an access mechanism, never as gearing).

**This is your call, not mine. But nothing should be built on the CFD path until it is made,
and the cautious answer is also the cheapest one: it deletes most of the remaining work.**

## Blocker (b): forex/CFD backtest data — options, ranked

Verified 2026-07-25.

1. **Polygon / Massive "Currencies" plan — CHECK THIS FIRST.** Same vendor already in use, so
   `data.py` already speaks the aggregates shape; forex and crypto are licensed together.
   Reported as a free Basic tier (5 calls/min, ~2 years history, reference + aggregate bars)
   with paid tiers above it. **UNVERIFIED and important:** whether MINUTE aggregates are on
   the free tier, or only end-of-day — the pricing page renders its Currencies tab
   client-side and could not be read programmatically. This project backtests on minute bars,
   so that one fact decides whether this option is nearly-free or costs money.
   **Action: open massive.com/pricing → Currencies tab and read off the minute-aggregate and
   history rows.** One page, no code, and it determines everything below.
2. **Dukascopy** — free, genuinely deep (tick through monthly, history to ~2003–2007), with a
   maintained `dukascopy-python` library. The best free option for forex depth. Cost: a
   second data path with a different shape, plus a new dependency.
3. **OANDA practice API** — real FX data, long daily history, another integration to write.
4. **IG's own historical prices — NOT a backtesting source.** Worth recording as a dead end
   so nobody tries it: IG's API has a finite **weekly datapoint allowance** and stores only
   limited history, and the consensus (including IG's own Labs material) is that it is
   insufficient for systematic backtesting. The tempting shortcut of "just use the broker's
   own data" does not work here.

## Crypto data — Binance public API (from the #36 source)

Answers the separately-noted "no crypto backtest data" gap behind the already-linked (and
deliberately empty) Coinbase/Kraken accounts.

- `GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m` — **no API key, no
  account, no signature.** 1000 candles per request; paginate with `startTime`/`endTime`.
  Intervals 1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.
- Caveats to carry into any build:
  - Binance quotes in **USDT**, not USD (`BTCUSDT`). Coinbase/Kraken USD pairs are close but
    not identical; one exchange's candles are not the other's fills.
  - Public market data is readable regardless of regional trading restrictions, but if a
    request ever returns HTTP 451 that is a geo-block, not a bug.
- **Not built, and shouldn't be yet.** Crypto needs its own strategy validation pass first —
  the GLD result (only 2/18 strategies profitable, Bollinger Mean Reversion going from +10.76%
  on equities to -1.66% on gold) is direct evidence that equity-tuned strategies do not
  transfer across instrument classes. Crypto is further away than gold was.
- 24/7 markets also break assumptions baked into the engine: `session_dates`, the
  `periods_per_year=252` default in metrics, `time_in_market` denominators, the market-hours
  guard, and the FOMC event calendar all assume a weekday session model. `volatility.py`
  already supports `periods_per_year=365`, which is the only piece currently ready.

## Revised sequence

1. **Read the Currencies pricing tab** (5 minutes, decides the data path).
2. **Make the leverage call** — recommendation above is to decline it.
3. If forex is still wanted after 1–2: **extend the IBKR adapter to spot FX**, and only then
   revisit IG if spread betting specifically matters.
4. Validate strategies on the new instrument class before any paper auto-trading, exactly as
   gold was validated (and rejected).
