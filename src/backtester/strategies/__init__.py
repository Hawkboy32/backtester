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
from backtester.strategies.pivot_point_scalping import PivotPointScalpingStrategy
from backtester.strategies.rsi_divergence import RsiDivergenceStrategy
from backtester.strategies.rsi_mean_reversion import RsiMeanReversionStrategy
from backtester.strategies.sma_crossover import SmaCrossoverStrategy
from backtester.strategies.volume_spike_reversal import VolumeSpikeReversalStrategy
from backtester.strategies.vwap_mean_reversion import VwapMeanReversionStrategy
from backtester.strategies.vwap_trend import VwapTrendStrategy

STRATEGY_REGISTRY: dict[str, dict] = {
    "SMA Crossover": {
        "class": SmaCrossoverStrategy,
        "default_params": {"fast_window": 20, "slow_window": 50},
    },
    "RSI Mean-Reversion": {
        "class": RsiMeanReversionStrategy,
        "default_params": {"period": 14, "oversold": 30.0, "overbought": 70.0},
    },
    "Bollinger Breakout": {
        "class": BollingerBreakoutStrategy,
        "default_params": {"period": 20, "num_std": 2.0},
    },
    "MACD Crossover": {
        "class": MacdCrossoverStrategy,
        "default_params": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
    },
    "Momentum ROC": {
        "class": MomentumRocStrategy,
        "default_params": {"period": 10, "buy_threshold": 0.0, "sell_threshold": 0.0},
    },
    "Liquidity Sweep": {
        "class": LiquiditySweepStrategy,
        "default_params": {"swing_window": 20},
    },
    "Volume Spike Reversal": {
        "class": VolumeSpikeReversalStrategy,
        "default_params": {"volume_window": 20, "volume_multiple": 2.0, "close_position_threshold": 0.6},
    },
    "Flag/Pennant Continuation": {
        "class": FlagPennantContinuationStrategy,
        "default_params": {
            "flagpole_window": 15,
            "consolidation_window": 8,
            "min_flagpole_move_pct": 3.0,
            "max_consolidation_range_pct": 40.0,
        },
    },
    "Multi-Timeframe Trend Pullback (EMA)": {
        "class": MultiTimeframePullbackStrategy,
        "default_params": {"trend_span": 50, "pullback_span": 20},
    },
    "VWAP Trend Following": {
        "class": VwapTrendStrategy,
        "default_params": {"min_bars": 5},
    },
    "VWAP Mean Reversion": {
        "class": VwapMeanReversionStrategy,
        "default_params": {"min_bars": 5, "entry_deviation_pct": 0.5},
    },
    "EMA Crossover Momentum": {
        "class": EmaCrossoverStrategy,
        "default_params": {"fast_span": 12, "slow_span": 26},
    },
    "Opening Range Breakout": {
        "class": OpeningRangeBreakoutStrategy,
        "default_params": {"opening_minutes": 15},
    },
    "Break-and-Retest": {
        "class": BreakAndRetestStrategy,
        "default_params": {"swing_window": 20, "retest_lookback": 15, "retest_tolerance_pct": 0.3},
    },
    "Bollinger Mean Reversion": {
        "class": BollingerMeanReversionStrategy,
        "default_params": {"period": 20, "num_std": 2.0},
    },
    "RSI Divergence": {
        "class": RsiDivergenceStrategy,
        "default_params": {"rsi_period": 14, "lookback": 10, "overbought": 70.0},
    },
    "Pivot Point Scalping": {
        "class": PivotPointScalpingStrategy,
        "default_params": {},
    },
    "Fibonacci Pullback": {
        "class": FibonacciPullbackStrategy,
        "default_params": {"swing_window": 30, "level": "0.618", "tolerance_pct": 0.3},
    },
}


def build_strategy(name: str, params: dict | None = None):
    entry = STRATEGY_REGISTRY[name]
    merged_params = {**entry["default_params"], **(params or {})}
    return entry["class"](**merged_params)
