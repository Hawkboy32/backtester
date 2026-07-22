"""Walk-forward GARCH(1,1) volatility forecasting and vol-targeted position sizing.

Ported from https://github.com/Miles-Deutscher/Garch-Method (same walk-forward
GARCH(1,1) construction, same regime percentile thresholds, same
target_vol/forecast_vol sizing formula) and adapted to this project's
conventions: DatetimeIndex OHLCV bars instead of a "date" column, exceptions
instead of sys.exit, and a position-size cap of 1.0x (never above 100% of
capital) rather than the original's 2.0x — this project has no margin/leverage
anywhere else, and vol-targeting shouldn't be the one place that introduces it.
GARCH forecasts the MAGNITUDE of the next day's move, not its direction.

Needs at least MIN_TRAIN + 10 days of DAILY closes to produce a first forecast
— this is a fundamentally different data requirement than the minute-bar
windows the rest of the bot backtests over, which is why this module fetches
and caches its own daily-bar series per ticker. MIN_TRAIN is set to 400
(not the original project's 500) because this account's Polygon plan only
returns ~2 years (~501 trading days) of daily history — 500+10 wouldn't fit at
all; 400 leaves enough trailing days for both a real forecast and a
percentile-based regime label to cover the bot's own (much shorter) scan/
backtest windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_TRAIN = 400          # days of daily history before the first forecast
REFIT_EVERY = 21         # re-estimate GARCH params every N days (walk-forward)
REGIME_LOOKBACK = 365    # window for the vol percentile / regime classification
REGIME_MIN_PERIODS = 30  # min forecast days before a percentile/regime is labeled
                         # (lower than a "real" year-long lookback would use, but
                         # with only ~500 days of daily history available in total
                         # (MIN_TRAIN=400 already spent), the original project's
                         # min_periods=90 left almost no labeled days to overlap
                         # with this bot's ~1-month scan/backtest windows)
DEFAULT_PERIODS_PER_YEAR = 252  # trading days/year; use 365 for a crypto-only ticker

DEFAULT_TARGET_VOL_ANN = 20.0  # annualized %, user-adjustable
MIN_SIZE = 0.25
MAX_SIZE = 1.0  # capped at "all-in", not 2.0x — this project takes no leverage anywhere

HONESTY_NOTE = (
    "GARCH forecasts magnitude (volatility), not direction. It tells you how "
    "violent tomorrow is likely to be — not which way it goes."
)


class InsufficientHistoryError(Exception):
    """Not enough daily bars to fit a walk-forward GARCH model."""


@dataclass
class RegimeInfo:
    date: object
    fcast_vol_daily_pct: float
    fcast_vol_ann_pct: float
    vol_pctile: float | None
    regime: str  # "calm" / "normal" / "storm"
    size_multiplier: float


def size_from_vol(
    forecast_vol_ann: float,
    target_vol_ann: float = DEFAULT_TARGET_VOL_ANN,
    max_size: float = MAX_SIZE,
    min_size: float = MIN_SIZE,
) -> float:
    """Position size multiplier from an annualized vol forecast (%).

    size = target_vol / forecast_vol, capped to [min_size, max_size]. Storm
    coming -> smaller position. Calm ahead -> bigger position (but never above
    max_size, i.e. never more than all-in).
    """
    if forecast_vol_ann is None or forecast_vol_ann <= 0 or np.isnan(forecast_vol_ann):
        return min_size
    return float(np.clip(target_vol_ann / forecast_vol_ann, min_size, max_size))


def walkforward_garch(
    daily_closes: pd.Series,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    min_train: int = MIN_TRAIN,
    refit_every: int = REFIT_EVERY,
) -> pd.DataFrame:
    """Walk-forward GARCH(1,1), zero lookahead.

    For each day t >= min_train, forecasts the vol of day t+1 using only data
    available at the close of day t. Params are re-estimated every
    `refit_every` days on an expanding window; between refits the GARCH
    recursion rolls forward with the last fitted params.

    `daily_closes` must be a Series of daily close prices indexed by date
    (ascending, no gaps required). Returns a DataFrame indexed like
    `daily_closes` (minus the first day) with: ret, fcast_vol (daily %),
    fcast_vol_ann (annualized %), vol_pctile, regime.
    """
    from arch import arch_model

    px = daily_closes.to_numpy(dtype=float)
    rets = 100.0 * np.diff(px) / px[:-1]
    n = len(rets)
    if n < min_train + 10:
        raise InsufficientHistoryError(
            f"Need at least {min_train + 10} days of daily closes; got {n + 1}."
        )

    fcast_var = np.full(n, np.nan)
    omega = alpha = beta = mu = None
    sigma2 = None

    for t in range(min_train, n):
        if (t - min_train) % refit_every == 0:
            am = arch_model(rets[:t], vol="GARCH", p=1, q=1, mean="Constant", dist="t")
            res = am.fit(disp="off", show_warning=False)
            p = res.params
            mu, omega, alpha, beta = p["mu"], p["omega"], p["alpha[1]"], p["beta[1]"]
            sigma2 = float(res.conditional_volatility[-1] ** 2)
        eps = rets[t] - mu
        sigma2 = omega + alpha * eps**2 + beta * sigma2
        fcast_var[t] = sigma2

    out = pd.DataFrame(index=daily_closes.index[1:])
    out["ret"] = rets
    out["fcast_vol"] = np.sqrt(fcast_var)
    out["fcast_vol_ann"] = out["fcast_vol"] * np.sqrt(periods_per_year)
    pct = out["fcast_vol"].rolling(REGIME_LOOKBACK, min_periods=REGIME_MIN_PERIODS).apply(
        lambda w: (w.iloc[:-1] < w.iloc[-1]).mean() * 100 if len(w) > 1 else np.nan, raw=False
    )
    out["vol_pctile"] = pct
    out["regime"] = pd.cut(
        out["vol_pctile"], bins=[-1, 33, 67, 101], labels=["calm", "normal", "storm"]
    )
    return out


def compute_regime_table(
    daily_bars: pd.DataFrame,
    target_vol_ann: float = DEFAULT_TARGET_VOL_ANN,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Daily-bar OHLCV DataFrame -> DataFrame indexed by date with fcast_vol_ann,
    vol_pctile, regime, size_multiplier. Ready to be forward-mapped onto a
    minute/hour/day bar series by calendar date.
    """
    res = walkforward_garch(daily_bars["close"], periods_per_year=periods_per_year)
    res["size_multiplier"] = res["fcast_vol_ann"].apply(
        lambda v: size_from_vol(v, target_vol_ann=target_vol_ann)
    )
    return res


def regime_by_date(table: pd.DataFrame) -> dict:
    """Convert a compute_regime_table() result into {date: {"regime": ..., "size_multiplier": ...}}
    for fast per-bar lookup (e.g. from BacktestEngine)."""
    out = {}
    for ts, row in table.iterrows():
        if pd.isna(row["regime"]):
            continue
        out[pd.Timestamp(ts).date()] = {
            "regime": str(row["regime"]),
            "size_multiplier": float(row["size_multiplier"]),
        }
    return out


def latest_regime_info(table: pd.DataFrame) -> RegimeInfo | None:
    """The most recent row with a valid forecast, for display/live-trading use."""
    valid = table.dropna(subset=["fcast_vol"])
    if valid.empty:
        return None
    latest = valid.iloc[-1]
    pctile = float(latest["vol_pctile"]) if pd.notna(latest["vol_pctile"]) else None
    return RegimeInfo(
        date=pd.Timestamp(valid.index[-1]).date(),
        fcast_vol_daily_pct=float(latest["fcast_vol"]),
        fcast_vol_ann_pct=float(latest["fcast_vol_ann"]),
        vol_pctile=pctile,
        regime=str(latest["regime"]) if pd.notna(latest["regime"]) else "normal",
        size_multiplier=float(latest["size_multiplier"]),
    )
