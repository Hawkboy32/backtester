"""Registry of available strategies, keyed by a stable display name."""

from __future__ import annotations

from backtester.strategies.bollinger_breakout import BollingerBreakoutStrategy
from backtester.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from backtester.strategies.bollinger_squeeze_breakout import BollingerSqueezeBreakoutStrategy
from backtester.strategies.break_and_retest import BreakAndRetestStrategy
from backtester.strategies.dmi_adx_trend import DmiAdxTrendStrategy
from backtester.strategies.dmi_dpo_guard import DmiDpoGuardStrategy
from backtester.strategies.dpo_mean_reversion import DpoMeanReversionStrategy
from backtester.strategies.ema_crossover import EmaCrossoverStrategy
from backtester.strategies.ema_rsi_confirmation import EmaRsiConfirmationStrategy
from backtester.strategies.linear_regression_channel import LinearRegressionChannelStrategy
from backtester.strategies.fibonacci_pullback import FibonacciPullbackStrategy
from backtester.strategies.flag_pennant import FlagPennantContinuationStrategy
from backtester.strategies.liquidity_sweep import LiquiditySweepStrategy
from backtester.strategies.macd_crossover import MacdCrossoverStrategy
from backtester.strategies.momentum_roc import MomentumRocStrategy
from backtester.strategies.multi_timeframe_pullback import MultiTimeframePullbackStrategy
from backtester.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from backtester.strategies.opening_range_liquidity_reversal import OpeningRangeLiquidityReversalStrategy
from backtester.strategies.opening_spike_fade import OpeningSpikeFadeStrategy
from backtester.strategies.pivot_point_scalping import PivotPointScalpingStrategy
from backtester.strategies.rsi_divergence import RsiDivergenceStrategy
from backtester.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from backtester.strategies.sma_crossover import SmaCrossoverStrategy
from backtester.strategies.triple_ema_ribbon import TripleEmaRibbonStrategy
from backtester.strategies.volume_spike_reversal import VolumeSpikeReversalStrategy
from backtester.strategies.vwap_drift_pullback import VwapDriftPullbackStrategy
from backtester.strategies.vwap_mean_reversion import VwapMeanReversionStrategy
from backtester.strategies.vwap_trend import VwapTrendStrategy

# "regime" tags which market behaviour a strategy is built for, so the roster
# can surface (and optionally require) ticker<->strategy matches against the
# scanner's per-ticker efficiency ratio (see metrics.classify_ticker_regime):
#   "trend" — profits from sustained directional moves (breakouts, crossovers,
#             continuation patterns);
#   "range" — profits from prices snapping back inside a band (mean reversion,
#             reversal patterns).
# Tags are judgment calls from each strategy's entry logic, not measurements.
STRATEGY_REGISTRY: dict[str, dict] = {
    "SMA Crossover": {
        "class": SmaCrossoverStrategy,
        "default_params": {"fast_window": 20, "slow_window": 50},
        "regime": "trend",
    },
    "RSI Mean-Reversion": {
        "class": RsiMeanReversionStrategy,
        "default_params": {"period": 14, "oversold": 30.0, "overbought": 70.0},
        "regime": "range",
    },
    "Bollinger Breakout": {
        "class": BollingerBreakoutStrategy,
        "default_params": {"period": 20, "num_std": 2.0},
        "regime": "trend",
    },
    "MACD Crossover": {
        "class": MacdCrossoverStrategy,
        "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "regime": "trend",
    },
    "Momentum ROC": {
        "class": MomentumRocStrategy,
        "default_params": {"period": 10, "buy_threshold": 0.0, "sell_threshold": 0.0},
        "regime": "trend",
    },
    "Liquidity Sweep": {
        "class": LiquiditySweepStrategy,
        "default_params": {"swing_window": 20},
        "regime": "range",
    },
    "Volume Spike Reversal": {
        "class": VolumeSpikeReversalStrategy,
        "default_params": {"volume_window": 20, "volume_multiple": 2.0, "close_position_threshold": 0.6},
        "regime": "range",
    },
    "Flag/Pennant Continuation": {
        "class": FlagPennantContinuationStrategy,
        "default_params": {
            "flagpole_window": 15,
            "consolidation_window": 8,
            "min_flagpole_move_pct": 3.0,
            "max_consolidation_range_pct": 40.0,
        },
        "regime": "trend",
    },
    "Multi-Timeframe Trend Pullback (EMA)": {
        "class": MultiTimeframePullbackStrategy,
        "default_params": {"trend_span": 50, "pullback_span": 20},
        "regime": "trend",
    },
    "VWAP Trend Following": {
        "class": VwapTrendStrategy,
        "default_params": {"min_bars": 5},
        "regime": "trend",
    },
    "VWAP Mean Reversion": {
        "class": VwapMeanReversionStrategy,
        "default_params": {"min_bars": 5, "entry_deviation_pct": 0.3},
        "regime": "range",
    },
    # NOT live-deployed — a testing-only variant for the "wait for the turn"
    # entry-confirmation investigation (2026-08-27/28, see
    # woolly-zooming-kurzweil.md). Same class, same tuned entry threshold,
    # confirm_turn_bars turned on. Never mutate the entry above; add a
    # sweep/roster/extra_targets entry pointed at THIS name to test it.
    "VWAP Mean Reversion (Turn-Confirmed)": {
        "class": VwapMeanReversionStrategy,
        "default_params": {"min_bars": 5, "entry_deviation_pct": 0.3, "confirm_turn_bars": 2},
        "regime": "range",
    },
    "VWAP Drift Pullback": {
        "class": VwapDriftPullbackStrategy,
        "default_params": {
            "min_bars": 5,
            "vwap_slope_lookback_bars": 15,
            "momentum_lookback_bars": 60,
            "momentum_threshold_pct": 0.1,
            "skip_opening_minutes": 60,
        },
        "regime": "trend",
    },
    "EMA Crossover Momentum": {
        "class": EmaCrossoverStrategy,
        "default_params": {"fast_span": 12, "slow_span": 26},
        "regime": "trend",
    },
    "Opening Range Breakout": {
        "class": OpeningRangeBreakoutStrategy,
        "default_params": {"opening_minutes": 15},
        "regime": "trend",
    },
    "Opening Range Liquidity Reversal": {
        "class": OpeningRangeLiquidityReversalStrategy,
        "default_params": {
            "opening_minutes": 15,
            "reversal_window_minutes": 90,
            "liquidity_multiplier": 1.5,
            "lookback_sessions": 5,
        },
        "regime": "range",
    },
    "Opening Spike Fade": {
        "class": OpeningSpikeFadeStrategy,
        "default_params": {
            "opening_minutes": 15,
            "reversal_window_minutes": 60,
            "min_move_pct": 0.1,
        },
        "regime": "range",
    },
    "Break-and-Retest": {
        "class": BreakAndRetestStrategy,
        "default_params": {"swing_window": 20, "retest_lookback": 15, "retest_tolerance_pct": 0.3},
        "regime": "trend",
    },
    "Bollinger Mean Reversion": {
        "class": BollingerMeanReversionStrategy,
        "default_params": {"period": 15, "num_std": 3.0},
        "regime": "range",
    },
    # NOT live-deployed — see "VWAP Mean Reversion (Turn-Confirmed)" above,
    # same reasoning, same investigation.
    "Bollinger Mean Reversion (Turn-Confirmed)": {
        "class": BollingerMeanReversionStrategy,
        "default_params": {"period": 15, "num_std": 3.0, "confirm_turn_bars": 2},
        "regime": "range",
    },
    "RSI Divergence": {
        "class": RsiDivergenceStrategy,
        "default_params": {"rsi_period": 14, "lookback": 10, "overbought": 70.0},
        "regime": "range",
    },
    "Pivot Point Scalping": {
        "class": PivotPointScalpingStrategy,
        "default_params": {},
        "regime": "range",
    },
    "Fibonacci Pullback": {
        "class": FibonacciPullbackStrategy,
        "default_params": {"swing_window": 30, "level": "0.618", "tolerance_pct": 0.3},
        "regime": "trend",
    },
    # Sourced from AlphaInsider strategy-browsing (2026-09-06) as candidates
    # worth testing - all three implement standard, textbook indicator
    # formulas (Wilder DMI/ADX 1978, Detrended Price Oscillator, Linear
    # Regression Channel), not any one script author's proprietary logic. Not
    # live-deployed by default - added to STRATEGY_REGISTRY so they CAN be
    # scanned/backtested and considered by the roster's normal rescan/
    # promotion process, same as every other candidate; nothing here forces
    # them into a live roster.
    "DMI/ADX Trend": {
        "class": DmiAdxTrendStrategy,
        "default_params": {"period": 14, "adx_threshold": 25.0},
        "regime": "trend",
    },
    "Linear Regression Channel": {
        "class": LinearRegressionChannelStrategy,
        # num_std widened 2.0 -> 3.0 after a smoke test showed the textbook
        # default overtrading badly on minute bars (394-498 trades/ticker in
        # 30 days, vs Bollinger Mean Reversion's own 26-40 at its tuned
        # num_std=3.0 on the same sample) - matching Bollinger's already-
        # learned lesson about band width on this project's actual bar
        # granularity, not the textbook value written for daily-bar use.
        "default_params": {"period": 20, "num_std": 3.0},
        "regime": "range",
    },
    "DPO Mean-Reversion": {
        "class": DpoMeanReversionStrategy,
        # Same overtrading finding as Linear Regression Channel above (614-699
        # trades/ticker at num_std=1.5) - widened to match.
        "default_params": {"period": 20, "num_std": 3.0},
        "regime": "range",
    },
    # Multi-indicator confirmation combos, built 2026-09-27 after every
    # single-signal strategy tested against the Kraken Funded challenge
    # sizing sweep (VWAP Mean Reversion, Momentum ROC, MACD Crossover,
    # Linear Regression Channel) scored 0-9% real pass rates - requiring a
    # second, independent signal to agree is meant to cut the false-signal
    # rate a lone indicator throws. DMI/DPO Guard sourced from AlphaInsider
    # (2026-09-27); the other two are standard, well-known combo techniques
    # (RSI-confirmed crossover, triple-EMA ribbon, TTM Squeeze-style
    # volatility breakout), not any one script author's proprietary logic.
    # Not live-deployed by default, same as every other candidate here.
    #
    # All four widened well past their textbook/daily-bar starting points
    # after a smoke test on 30 days of real 1-min BTC data caught the SAME
    # overtrading problem Linear Regression Channel/DPO Mean-Reversion
    # already hit once (350-1060 trades/30-days, equity down 29-56% purely
    # from cumulative slippage) - re-tuned by measuring trade count/equity
    # across widened parameter sets on the same sample, not guessed.
    "DMI/DPO Guard": {
        "class": DmiDpoGuardStrategy,
        # dpo_guard_std 0.5->2.0: 1060 trades/30d -> 96, equity 444->940.
        "default_params": {"dmi_period": 14, "dpo_period": 20, "dpo_guard_std": 2.0},
        "regime": "trend",
    },
    "EMA/RSI Confirmation": {
        "class": EmaRsiConfirmationStrategy,
        # spans 12/26->100/300: 772 trades/30d -> 64, equity 494->936 - the
        # best equity outcome of all four smoke-tested candidates.
        "default_params": {"fast_span": 100, "slow_span": 300, "rsi_period": 14, "rsi_midline": 50.0},
        "regime": "trend",
    },
    "Triple EMA Ribbon": {
        "class": TripleEmaRibbonStrategy,
        # spans 9/21/55->50/150/300: 793 trades/30d -> 122, equity 471->891.
        "default_params": {"fast_span": 50, "mid_span": 150, "slow_span": 300},
        "regime": "trend",
    },
    "Bollinger Squeeze Breakout": {
        "class": BollingerSqueezeBreakoutStrategy,
        # num_std 2.0->3.0 (matches Bollinger Mean Reversion's own tuned
        # value), squeeze_tolerance 1.10->1.05 (stricter): 354 trades/30d ->
        # 72, equity 714->923.
        "default_params": {
            "period": 20, "num_std": 3.0, "squeeze_lookback": 50, "squeeze_tolerance": 1.05,
            "squeeze_recent_bars": 5, "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        },
        "regime": "trend",
    },
}


def strategy_regime(name: str) -> str:
    """The regime tag ("trend"/"range") for a registered strategy; "either"
    for anything unregistered or untagged (never blocks a match)."""
    return STRATEGY_REGISTRY.get(name, {}).get("regime", "either")


def build_strategy(name: str, params: dict | None = None):
    entry = STRATEGY_REGISTRY[name]
    merged_params = {**entry["default_params"], **(params or {})}
    return entry["class"](**merged_params)
