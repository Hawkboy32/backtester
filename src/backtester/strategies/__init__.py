"""Registry of available strategies, keyed by a stable display name."""

from __future__ import annotations

from backtester.strategies.bollinger_breakout import BollingerBreakoutStrategy
from backtester.strategies.bollinger_mean_reversion import BollingerMeanReversionStrategy
from backtester.strategies.break_and_retest import BreakAndRetestStrategy
from backtester.strategies.ema_crossover import EmaCrossoverStrategy
from backtester.strategies.fibonacci_pullback import FibonacciPullbackStrategy
from backtester.strategies.flag_pennant import FlagPennantContinuationStrategy
from backtester.strategies.liquidity_sweep import LiquiditySweepStrategy
from backtester.strategies.macd_crossover import MacdCrossoverStrategy
from backtester.strategies.momentum_roc import MomentumRocStrategy
from backtester.strategies.multi_timeframe_pullback import MultiTimeframePullbackStrategy
from backtester.strategies.opening_range_breakout import OpeningRangeBreakoutStrategy
from backtester.strategies.opening_range_liquidity_reversal import OpeningRangeLiquidityReversalStrategy
from backtester.strategies.pivot_point_scalping import PivotPointScalpingStrategy
from backtester.strategies.rsi_divergence import RsiDivergenceStrategy
from backtester.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from backtester.strategies.sma_crossover import SmaCrossoverStrategy
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
}


def strategy_regime(name: str) -> str:
    """The regime tag ("trend"/"range") for a registered strategy; "either"
    for anything unregistered or untagged (never blocks a match)."""
    return STRATEGY_REGISTRY.get(name, {}).get("regime", "either")


def build_strategy(name: str, params: dict | None = None):
    entry = STRATEGY_REGISTRY[name]
    merged_params = {**entry["default_params"], **(params or {})}
    return entry["class"](**merged_params)
