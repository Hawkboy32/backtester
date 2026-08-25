"""GBP capital-gains estimate per account, from REALIZED trade P&L only
(live_trades.db) — not unrealized/open-position P&L, matching how UK CGT
actually treats a position (a gain only exists once you've sold).

DELIBERATELY THE SIMPLE VERSION, not full HMRC Section 104 share-pooling
(2026-08-17, user's own choice after being shown the tradeoff): this sums
each account's realized P&L for the tax year and treats that as the gain.
Real UK CGT pools same-ticker buys into an average cost basis and applies
same-day/30-day "bed and breakfast" matching rules, which this does NOT
reproduce — for an account trading the same ticker repeatedly with
overlapping buys, the two methods can give different numbers. This is a
running ESTIMATE for your own tracking, not a substitute for an actual
Self Assessment calculation.

RATES/ALLOWANCE ARE NOT HARDCODED ON PURPOSE. UK CGT allowance and rate(s)
change with the budget and this assistant has no reliable way to know
today's actual figures — TaxSettings defaults to 0.0 for both, forcing an
explicit real value from the user (via the dashboard/mobile settings UI)
rather than silently computing against a number that might be stale or
simply wrong. Same reasoning for the GBP/USD rate: manually entered, not
fetched, so accuracy is exactly as good as how recently it was updated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from backtester.auto_trader_state import STATE_DIR, atomic_write_text, read_state_json
from backtester.live_trades import realized_pnl_by_account_between

PATH = STATE_DIR / "tax_settings.json"

# Sensible starting guesses only, NOT asserted as correct - both are exactly
# as editable as every other entry in account_currencies. Only an account
# already confirmed GBP-denominated (IBKR Live, 2026-08-16) defaults to GBP;
# everything else defaults to USD until the user says otherwise. Paper
# accounts (e.g. MyIGPaper) never reach this lookup at all - compute_tax_
# summary filters to live accounts before it's ever called - so there's no
# reason to pre-seed a paper account's id here even as a placeholder.
_DEFAULT_GBP_ACCOUNT_IDS = {
    "e7181ec2-e486-4cef-9324-43895cbb7b8d",  # IBKR Live
}


@dataclass
class TaxSettings:
    cgt_allowance_gbp: float = 0.0
    cgt_rate_pct: float = 0.0
    gbp_usd_rate: float = 0.0  # USD per 1 GBP, e.g. 1.27
    account_currencies: dict[str, str] = field(default_factory=dict)  # account_id -> "GBP"/"USD"

    def currency_for(self, account_id: str) -> str:
        if account_id in self.account_currencies:
            return self.account_currencies[account_id]
        return "GBP" if account_id in _DEFAULT_GBP_ACCOUNT_IDS else "USD"


def load_tax_settings() -> TaxSettings:
    data = read_state_json(PATH, default={})
    return TaxSettings(
        cgt_allowance_gbp=data.get("cgt_allowance_gbp", 0.0),
        cgt_rate_pct=data.get("cgt_rate_pct", 0.0),
        gbp_usd_rate=data.get("gbp_usd_rate", 0.0),
        account_currencies=data.get("account_currencies", {}),
    )


def save_tax_settings(settings: TaxSettings) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(PATH, json.dumps({
        "cgt_allowance_gbp": settings.cgt_allowance_gbp,
        "cgt_rate_pct": settings.cgt_rate_pct,
        "gbp_usd_rate": settings.gbp_usd_rate,
        "account_currencies": settings.account_currencies,
    }, indent=2))


def uk_tax_year_bounds(as_of: date | None = None) -> tuple[date, date]:
    """UK tax year runs 6 April to 5 April inclusive - a fixed structural
    fact (unlike the rate/allowance above), not something that needs a
    user-editable setting. Returns (start, end) for the tax year containing
    as_of (today if not given)."""
    d = as_of or date.today()
    year_start = d.year if (d.month, d.day) >= (4, 6) else d.year - 1
    return date(year_start, 4, 6), date(year_start + 1, 4, 5)


@dataclass
class AccountTaxLine:
    account_id: str
    nickname: str
    currency: str
    realized_gain_native: float
    realized_gain_gbp: float | None  # None when a USD account has no FX rate set yet


@dataclass
class TaxSummary:
    tax_year_start: date
    tax_year_end: date
    lines: list[AccountTaxLine]
    total_gain_gbp: float  # sum over lines with a known GBP figure only
    missing_fx_accounts: list[str]  # nicknames excluded from the total for lack of an FX rate
    allowance_gbp: float
    taxable_gain_gbp: float
    rate_pct: float
    estimated_tax_gbp: float


def compute_tax_summary(accounts: list[dict], as_of: date | None = None) -> TaxSummary:
    """accounts: the same list `backtester.accounts.list_accounts()` returns
    (needs id/nickname/is_paper) - passed in rather than imported so this
    module doesn't need broker-credential access just to do arithmetic.

    Filtered to LIVE (real-money, is_paper=False) accounts only (2026-08-17,
    user's own call) - paper accounts have no real tax exposure, so
    including them just adds noise to a figure meant for real tracking. A
    caller that only ever passes live accounts sees no change from this."""
    settings = load_tax_settings()
    start, end = uk_tax_year_bounds(as_of)
    pnl_by_account = realized_pnl_by_account_between(start.isoformat(), (end.isoformat() + "T23:59:59"))
    live_accounts = [a for a in accounts if not a.get("is_paper", True)]

    lines: list[AccountTaxLine] = []
    missing_fx: list[str] = []
    total_gbp = 0.0
    for acct in live_accounts:
        native = pnl_by_account.get(acct["id"], 0.0)
        currency = settings.currency_for(acct["id"])
        if currency == "GBP":
            gbp = native
        elif settings.gbp_usd_rate > 0:
            gbp = native / settings.gbp_usd_rate
        else:
            gbp = None
            if native != 0.0:
                missing_fx.append(acct["nickname"])
        if gbp is not None:
            total_gbp += gbp
        lines.append(AccountTaxLine(
            account_id=acct["id"], nickname=acct["nickname"], currency=currency,
            realized_gain_native=native, realized_gain_gbp=gbp,
        ))

    taxable = max(0.0, total_gbp - settings.cgt_allowance_gbp)
    estimated_tax = taxable * settings.cgt_rate_pct / 100

    return TaxSummary(
        tax_year_start=start, tax_year_end=end, lines=lines, total_gain_gbp=total_gbp,
        missing_fx_accounts=missing_fx, allowance_gbp=settings.cgt_allowance_gbp,
        taxable_gain_gbp=taxable, rate_pct=settings.cgt_rate_pct, estimated_tax_gbp=estimated_tax,
    )
