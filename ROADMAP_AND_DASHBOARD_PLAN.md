> **STATUS (2026-07-26): this plan is done.** Reorg, IBKR, conviction logging, and the
> news panel all shipped 2026-07-25. Kept here as historical record only — for the
> CURRENT roadmap (crypto validation -> unleveraged forex via IBKR -> a dedicated IG/
> leverage discussion), see the "crypto + forex/CFD action plan" entry near the end of
> CLAUDE_NOTES.txt.

# Roadmap + dashboard simplification plan (draft, 2026-07-21)

_A plan for the ORDER we tackle the pending additions, what each does to the dashboard,
and how to simplify the dashboard to reduce confusion. Draft for the user to react to —
nothing here is built yet._

## Why simplify now
The dashboard started as 7 flat tabs and has accreted a lot: GARCH toggles, adaptive
roster, account-risk breaker, notifications, market-status, live P&L, sizing presets,
cost presets. Symptoms of drift:
- **Settings are scattered** — Polygon key + notifications on API Keys; execution-cost
  defaults in the Backtest/Scanner sidebars; account-risk + live-arm phrase buried in Auto
  Trading.
- **Redundancy** — positions render in both Accounts and Trade Execution.
- **The Auto Trading tab is overloaded** — mode toggle, roster config, vol targeting,
  account risk, arm phrase, start/stop, event log — all on one screen.
- **No single "daily glance" view** — the daily health check is currently pieced together
  from Auto Trading + Accounts + the phone.
- **IG will add the most UI ever** (CFD/forex: instrument types, leverage caps, a new
  universe). Doing that on top of the current sprawl would compound confusion.

**Key principle: simplify the dashboard BEFORE adding IG. Do IBKR either before or after
the reorg (low UI impact). This keeps the most complex addition landing on a clean base.**

## Proposed build order
1. **Thursday: validation review + tiny #26 fix.** Review the paper run's results; batch the
   BUY-notification qty fix (#26) with the restart (it needs one anyway). Decide whether to
   continue the run / go again.
2. **Dashboard simplification / reorg.** Foundational — reduces confusion now AND gives
   every later addition a clean home. (Detail below.)
3. **IBKR integration (#30).** Clean: real US stocks reuse the existing equities data path.
   Lands in the simplified Accounts/Trading structure. Longevity foundation.
4. **Research enhancements as they fit:** conviction sizing measurement (#25 — start by just
   LOGGING conviction, no sizing yet) and the news context panel (#21, read-only). Both slot
   into the simplified structure without much new surface.
5. **Forex/CFD data source + leverage decision → IG integration (#31).** The big one. Only
   after its two gates are cleared; benefits most from the simplified dashboard.
6. **Pre-remote-access hardening → Pi/VPN move:** 2FA (#22) + keyring backend swap, then the
   Raspberry Pi + Tailscale hosting. Gated on the bot proving net-positive on paper first.

## What each addition does to the dashboard
| Addition | Dashboard impact |
|---|---|
| #26 notification qty fix | None (backend only) |
| Dashboard reorg | Restructures navigation; net SIMPLER (see below) |
| #30 IBKR | +1 broker in the Accounts link flow; an "IBKR gateway running?" status chip. Reuses existing Accounts/Execution/Auto Trading. Low impact. |
| #25 conviction sizing | A sizing option/toggle (Auto Trading + Execution) and a conviction column in results. Small; phase it in as measure-first. |
| #21 news panel | One read-only panel (headlines per ticker) — lives in the Overview or a research view. Low impact. |
| #31 IG (CFD/forex) | HIGHEST impact: instrument-type concept (stocks vs CFDs vs forex), leverage-cap controls, a forex/CFD universe, contract/lot sizing vs shares. This is the reason to reorg first. |
| #22 2FA | Login-flow step + a Settings entry. |

## Dashboard simplification proposal
Move from 7 flat tabs to a **grouped navigation with a daily-glance home**, using
`st.navigation` / `st.Page` (sidebar sections). Proposed structure:

```
OVERVIEW  (new — the daily glance)
  • bot status (running/armed/trades today/last error)
  • open positions with live colour-coded P&L
  • recent trades + notification status
  • kill switch always reachable here

RESEARCH
  • Backtest
  • Strategy Scanner
  • Scan History
  • (later) News context

TRADING
  • Accounts        (link/remove, balances, market status — positions move to Overview)
  • Trade Execution (manual orders, sizing presets, close positions)
  • Auto Trading    (slimmed: mode + start/stop + roster table; heavy config behind an
                     "Advanced settings" expander)

SETTINGS  (one home for all config)
  • API keys (Polygon)
  • Notifications
  • Execution-cost defaults (the $0/2bps + large/small-cap presets — currently in sidebars)
  • Account risk defaults (max-drawdown breaker)
  • (later) 2FA
```

Concrete simplifications:
- **Add an Overview home** so the daily check is one screen, not three.
- **Consolidate all settings** into a Settings section (stop scattering them across sidebars
  + API Keys + Auto Trading).
- **De-duplicate positions** — live P&L positions live in Overview; Accounts keeps only
  link/balance management; Execution keeps only the close-position action.
- **Slim Auto Trading** — surface mode + start/stop + roster table; push roster thresholds,
  vol targeting, and account-risk knobs behind a clearly-labelled "Advanced settings"
  expander (progressive disclosure).
- **Kill switch** reachable from Overview (and Auto Trading), never buried.
- **Prepare for instrument types** — the reorg should leave an obvious place for IG's
  stocks-vs-CFDs-vs-forex distinction and leverage caps to slot in without another rethink.

## Decisions made (2026-07-21)
1. **Structure:** GROUPED NAV (st.navigation/st.Page sidebar sections) + a dedicated
   **Overview home** — confirmed. Overview / Research / Trading / Settings as laid out above.
2. **Overview home:** YES — build the daily-glance page (bot status, live P&L positions,
   recent trades, kill switch on one screen).
3. **Timing:** REORG FIRST, then IBKR, then IG. The simplification is the next build after
   Thursday's validation review (+ the tiny #26 notification fix batched with the restart).

## Confirmed sequence
Thursday review (+ #26) → **dashboard reorg (grouped nav + Overview home)** → IBKR (#30) →
research bits (#25 conviction logging, #21 news) → forex data + leverage decision → IG (#31)
→ 2FA (#22) + keyring swap → Pi/VPN hosting. (Pi gated on the bot proving net-positive on
paper.)
