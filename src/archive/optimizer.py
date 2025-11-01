"""
Parameter Optimization Module

This module provides fast grid search and random search capabilities to find
optimal trading parameters using parallel processing.
"""

import itertools
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src import config
from src.archive.fetch_historical import fetch_historical_prices
from src.archive.metrics import add_scalping_metrics, add_trade_tags
from src.simulator.sim_trades import simulate_trades
from src.evaluation.performance_metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_calmar_ratio
)
from logs import logger


class ParameterOptimizer:
    """
    Fast parameter optimizer using parallel processing.

    Optimizes trading strategy parameters efficiently by:
    - Using all available CPU cores by default
    - Testing focused parameter ranges for speed
    - Supporting both grid and random search
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        days: int = 14,
        granularity: str = "5m",
        start_cash: float = 100_000.0,
        optimization_metric: str = "sharpe_ratio",
        n_jobs: Optional[int] = None
    ):
        """
        Initialize the optimizer.

        Args:
            data: Pre-loaded DataFrame (if None, will fetch fresh data)
            days: Number of days of historical data to use
            granularity: Candle granularity (1m, 5m, 15m, 1h, etc.)
            start_cash: Starting capital for backtests
            optimization_metric: Metric to optimize ("sharpe_ratio", "total_return",
                                "win_rate", "sortino_ratio", "calmar_ratio", "profit_factor")
            n_jobs: Number of parallel jobs (None = use all CPU cores)
        """
        self.data = data
        self.days = days
        self.granularity = granularity
        self.start_cash = start_cash
        self.optimization_metric = optimization_metric
        self.n_jobs = n_jobs if n_jobs is not None else max(1, cpu_count() - 1)
        self.results = []

        logger.info(f"Optimizer initialized with {self.n_jobs} parallel workers")

    def define_parameter_space(
        self,
        preset: str = "fast"
    ) -> Dict[str, List[Any]]:
        """
        Define the parameter space to search.

        Args:
            preset: "fast" (~50 runs), "balanced" (~200 runs), or "thorough" (~500 runs)

        Returns:
            Dictionary of parameter names to lists of values to test
        """
        if preset == "fast":
            # Ultra-fast search (~30-60 combinations)
            return {
                # Bollinger Bands
                "bb_window": [20],
                "bb_stdev": [1.75, 2.0],

                # Moving Averages
                "sma_fast": [5, 8],
                "sma_slow": [20, 25],
                "ema_short": [12],
                "ema_long": [60],

                # RSI
                "rsi_period": [14],
                "rsi_low": [30],
                "rsi_high": [60],

                # Signal Thresholds
                "min_buy_votes": [2],
                "min_sell_votes": [2],

                # Risk Management (most important to optimize)
                "stop_loss_pct": [0.005, 0.0075, 0.01],
                "take_profit_pct": [0.01, 0.015, 0.02],
                "max_position_size": [0.3, 0.5],
                "position_sizing_method": ["fixed", "kelly"],
            }

        elif preset == "balanced":
            # Balanced search (~150-250 runs)
            return {
                # Bollinger Bands
                "bb_window": [14, 20],
                "bb_stdev": [1.5, 2.0, 2.5],

                # Moving Averages
                "sma_fast": [5, 8, 10],
                "sma_slow": [20, 25],
                "ema_short": [8, 12],
                "ema_long": [30, 60],

                # RSI
                "rsi_period": [7, 14],
                "rsi_low": [25, 30, 35],
                "rsi_high": [60, 70],

                # Signal Thresholds
                "min_buy_votes": [2, 3],
                "min_sell_votes": [2],

                # Risk Management
                "stop_loss_pct": [0.005, 0.0075, 0.01],
                "take_profit_pct": [0.01, 0.015, 0.02],
                "max_position_size": [0.3, 0.5],
                "position_sizing_method": ["fixed", "kelly"],

                # Advanced Features
                "trailing_stop_pct": [None, 0.01],
            }

        elif preset == "thorough":
            # More comprehensive search (~400-600 runs)
            return {
                # Bollinger Bands
                "bb_window": [14, 20, 25],
                "bb_stdev": [1.5, 1.75, 2.0, 2.5],

                # Moving Averages
                "sma_fast": [5, 8, 10],
                "sma_slow": [20, 25, 40],
                "ema_short": [8, 12, 15],
                "ema_long": [30, 60, 75],

                # RSI
                "rsi_period": [7, 14, 21],
                "rsi_low": [20, 30, 35],
                "rsi_high": [50, 60, 70],

                # Signal Thresholds
                "min_buy_votes": [2, 3],
                "min_sell_votes": [2, 3],

                # Risk Management
                "stop_loss_pct": [0.005, 0.0075, 0.01, 0.015],
                "take_profit_pct": [0.008, 0.01, 0.012, 0.015, 0.02],
                "max_position_size": [0.2, 0.3, 0.5],
                "position_sizing_method": ["fixed", "kelly", "volatility"],

                # Advanced Features
                "trailing_stop_pct": [None, 0.01, 0.015],
                "partial_profit_pct": [None, 0.005],
            }

        else:
            raise ValueError(f"Unknown preset: {preset}. Use 'fast', 'balanced', or 'thorough'")

    def _load_data(self) -> pd.DataFrame:
        """Load historical data if not already loaded."""
        if self.data is not None:
            return self.data

        logger.info(f"Fetching {self.days} days of {config.SYMBOL}/USDT data at {self.granularity}")

        # Extract minutes from granularity string (e.g., "5m" -> 5, "15m" -> 15)
        if self.granularity.endswith('m'):
            resample_minutes = int(self.granularity[:-1])
        elif self.granularity.endswith('h'):
            resample_minutes = int(self.granularity[:-1]) * 60
        else:
            resample_minutes = 5  # default

        df = fetch_historical_prices(
            days=self.days,
            resample_minutes=resample_minutes,
            exchange_id="binance"
        )
        return df

    def _run_single_backtest(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Run a single backtest with given parameters.

        Args:
            params: Parameter dictionary
            df: OHLCV DataFrame

        Returns:
            Dictionary with parameters and performance metrics
        """
        try:
            # Create a copy to avoid modifying original
            df_test = df.copy()

            # Calculate indicators with these parameters
            df_test = add_scalping_metrics(
                df_test,
                bb_window=params.get("bb_window", 20),
                bb_std=params.get("bb_stdev", 2.0),  # Note: function uses bb_std, not bb_stdev
                sma_fast=params.get("sma_fast", 8),
                sma_slow=params.get("sma_slow", 25),
                ema_short=params.get("ema_short", 12),
                ema_long=params.get("ema_long", 60),
                rsi_window=params.get("rsi_period", 14)  # Note: function uses rsi_window, not rsi_period
            )

            # Generate signals with these thresholds
            # First generate with default votes to get buy_score and sell_score
            df_test = add_trade_tags(
                df_test,
                rsi_low=params.get("rsi_low", 30),
                rsi_high=params.get("rsi_high", 60)
            )

            # Then regenerate buy/sell based on custom vote thresholds
            # (since add_trade_tags uses hardcoded MIN_BUY_VOTES from config)
            min_buy_votes = params.get("min_buy_votes", 2)
            min_sell_votes = params.get("min_sell_votes", 2)
            df_test["buy"] = df_test["buy_score"] >= min_buy_votes
            df_test["sell"] = df_test["sell_score"] >= min_sell_votes
            # Prevent simultaneous signals
            df_test["sell"] = df_test["sell"] & ~df_test["buy"]

            # Run simulation with these risk parameters
            df_result, trades_df, summary = simulate_trades(
                df_test,
                start_cash=self.start_cash,
                buy_fee=0.001,
                sell_fee=0.001,
                target_pct=params.get("take_profit_pct", 0.01),
                stop_loss_pct=params.get("stop_loss_pct", 0.0075),
                max_position_size=params.get("max_position_size", 0.3),
                position_sizing_method=params.get("position_sizing_method", "fixed"),
                trailing_stop_pct=params.get("trailing_stop_pct"),
                partial_profit_pct=params.get("partial_profit_pct"),
                partial_profit_size=params.get("partial_profit_size", 0.5),
                cooldown_bars=params.get("cooldown_bars", 2)
            )

            # Calculate performance metrics
            if len(trades_df) > 0:
                returns = df_result["equity"].pct_change().dropna()

                sharpe = calculate_sharpe_ratio(returns)
                sortino = calculate_sortino_ratio(returns)
                max_dd, _, _ = calculate_max_drawdown(df_result["equity"])  # Returns tuple: (max_dd, peak_idx, trough_idx)
                calmar = calculate_calmar_ratio(df_result["equity"])  # Pass equity series, not individual values

                # Profit factor = sum(wins) / abs(sum(losses))
                wins = trades_df[trades_df["net_ret"] > 0]["realized_pnl"].sum()
                losses = abs(trades_df[trades_df["net_ret"] < 0]["realized_pnl"].sum())
                profit_factor = wins / losses if losses > 0 else float('inf')

            else:
                sharpe = -999.0
                sortino = -999.0
                max_dd = 0.0
                calmar = -999.0
                profit_factor = 0.0

            # Compile results
            result = {
                **params,
                "total_return": summary["total_net_ret"],
                "win_rate": summary["win_rate"],
                "num_trades": summary["n_trades"],
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "calmar_ratio": calmar,
                "profit_factor": profit_factor,
                "final_equity": summary["end_equity"],
                "avg_return_per_trade": summary["avg_net_ret"]
            }

            return result

        except Exception as e:
            import traceback
            logger.error(f"Backtest failed with params {params}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                **params,
                "total_return": -1.0,
                "win_rate": 0.0,
                "num_trades": 0,
                "sharpe_ratio": -999.0,
                "sortino_ratio": -999.0,
                "max_drawdown": -1.0,
                "calmar_ratio": -999.0,
                "profit_factor": 0.0,
                "final_equity": 0.0,
                "avg_return_per_trade": -1.0,
                "error": str(e)
            }

    def grid_search(
        self,
        param_space: Optional[Dict[str, List[Any]]] = None,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Perform fast grid search over parameter space using parallel processing.

        Args:
            param_space: Parameter space dictionary (if None, uses fast preset)
            save_results: Whether to save results to CSV

        Returns:
            DataFrame with all parameter combinations and their performance
        """
        if param_space is None:
            param_space = self.define_parameter_space("fast")

        # Generate all combinations
        keys = list(param_space.keys())
        values = list(param_space.values())
        combinations = list(itertools.product(*values))

        logger.info(f"Starting grid search with {len(combinations)} parameter combinations")
        logger.info(f"Using {self.n_jobs} parallel workers")
        logger.info(f"Optimizing for: {self.optimization_metric}")

        # Load data once
        df = self._load_data()

        # Run backtests in parallel
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for combo in combinations:
                params = dict(zip(keys, combo))
                future = executor.submit(self._run_single_backtest, params, df)
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search"):
                results.append(future.result())

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Sort by optimization metric
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(config.PROJECT_ROOT, "data", "optimization")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"grid_search_{timestamp}.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved optimization results to {output_path}")

        self.results = results
        return results_df

    def random_search(
        self,
        param_space: Optional[Dict[str, List[Any]]] = None,
        n_iterations: int = 50,
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Perform fast random search using parallel processing.

        Args:
            param_space: Parameter space dictionary (if None, uses thorough preset)
            n_iterations: Number of random combinations to test
            save_results: Whether to save results to CSV

        Returns:
            DataFrame with sampled parameter combinations and their performance
        """
        if param_space is None:
            param_space = self.define_parameter_space("thorough")

        logger.info(f"Starting random search with {n_iterations} iterations")
        logger.info(f"Using {self.n_jobs} parallel workers")
        logger.info(f"Optimizing for: {self.optimization_metric}")

        # Load data once
        df = self._load_data()

        # Generate random combinations
        keys = list(param_space.keys())
        random_combos = []
        for _ in range(n_iterations):
            combo = {key: random.choice(param_space[key]) for key in keys}
            random_combos.append(combo)

        # Run backtests in parallel
        results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            futures = []
            for params in random_combos:
                future = executor.submit(self._run_single_backtest, params, df)
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="Random Search"):
                results.append(future.result())

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Sort by optimization metric
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)

        # Save results
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(config.PROJECT_ROOT, "data", "optimization")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"random_search_{timestamp}.csv")
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved optimization results to {output_path}")

        self.results = results
        return results_df

    def get_best_parameters(
        self,
        top_n: int = 5
    ) -> pd.DataFrame:
        """
        Get the top N best parameter combinations.

        Args:
            top_n: Number of top results to return

        Returns:
            DataFrame with top N parameter combinations
        """
        if not self.results:
            raise ValueError("No results available. Run optimization first.")

        results_df = pd.DataFrame(self.results)
        return results_df.nlargest(top_n, self.optimization_metric)

    def export_best_config(
        self,
        output_path: Optional[str] = None,
        rank: int = 1
    ) -> Dict[str, Any]:
        """
        Export the best parameter set as a config-compatible format.

        Args:
            output_path: Path to save JSON config (if None, just returns dict)
            rank: Which ranked result to export (1 = best, 2 = second best, etc.)

        Returns:
            Dictionary of best parameters
        """
        if not self.results:
            raise ValueError("No results available. Run optimization first.")

        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)

        if rank > len(results_df):
            raise ValueError(f"Rank {rank} exceeds number of results ({len(results_df)})")

        best_params = results_df.iloc[rank - 1].to_dict()

        # Format for config.py
        config_dict = {
            "BB_WINDOW": int(best_params["bb_window"]),
            "BB_STDEV": float(best_params["bb_stdev"]),
            "SMA_FAST": int(best_params["sma_fast"]),
            "SMA_SLOW": int(best_params["sma_slow"]),
            "EMA_SHORT": int(best_params["ema_short"]),
            "EMA_LONG": int(best_params["ema_long"]),
            "RSI": int(best_params["rsi_period"]),
            "RSI_LOW": int(best_params["rsi_low"]),
            "RSI_HIGH": int(best_params["rsi_high"]),
            "MIN_BUY_VOTES": int(best_params["min_buy_votes"]),
            "MIN_SELL_VOTES": int(best_params["min_sell_votes"]),
            "MAX_POSITION_SIZE": float(best_params["max_position_size"]),
            "POSITION_SIZING_METHOD": str(best_params["position_sizing_method"]),
            "STOP_LOSS_PCT": float(best_params["stop_loss_pct"]),
            "TAKE_PROFIT_PCT": float(best_params["take_profit_pct"]),
            "TRAILING_STOP_PCT": best_params.get("trailing_stop_pct"),
            "PARTIAL_PROFIT_PCT": best_params.get("partial_profit_pct"),
            "PARTIAL_PROFIT_SIZE": best_params.get("partial_profit_size", 0.5),
            "PERFORMANCE_METRICS": {
                "total_return": float(best_params["total_return"]),
                "win_rate": float(best_params["win_rate"]),
                "sharpe_ratio": float(best_params["sharpe_ratio"]),
                "sortino_ratio": float(best_params["sortino_ratio"]),
                "max_drawdown": float(best_params["max_drawdown"]),
                "num_trades": int(best_params["num_trades"])
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Exported best config to {output_path}")

        return config_dict


def optimize_parameters(
    days: int = 14,
    granularity: str = "5m",
    method: str = "grid",
    preset: str = "fast",
    n_iterations: int = 50,
    optimization_metric: str = "sharpe_ratio",
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to run fast parameter optimization.

    Args:
        days: Number of days of historical data
        granularity: Candle granularity
        method: "grid" or "random"
        preset: "fast", "balanced", or "thorough" (for grid search)
        n_iterations: Number of iterations (for random search)
        optimization_metric: Metric to optimize
        n_jobs: Number of parallel jobs (None = use all CPUs)

    Returns:
        Tuple of (results DataFrame, best parameters dict)
    """
    optimizer = ParameterOptimizer(
        days=days,
        granularity=granularity,
        optimization_metric=optimization_metric,
        n_jobs=n_jobs
    )

    if method == "grid":
        param_space = optimizer.define_parameter_space(preset)
        results_df = optimizer.grid_search(param_space)
    elif method == "random":
        results_df = optimizer.random_search(n_iterations=n_iterations)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'grid' or 'random'")

    # Get best parameters
    best_params = optimizer.export_best_config()

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*60)
    logger.info(f"\nBest Parameters (ranked by {optimization_metric}):")
    logger.info(f"  Total Return: {best_params['PERFORMANCE_METRICS']['total_return']*100:.2f}%")
    logger.info(f"  Win Rate: {best_params['PERFORMANCE_METRICS']['win_rate']*100:.1f}%")
    logger.info(f"  Sharpe Ratio: {best_params['PERFORMANCE_METRICS']['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {best_params['PERFORMANCE_METRICS']['max_drawdown']*100:.2f}%")
    logger.info(f"  Number of Trades: {best_params['PERFORMANCE_METRICS']['num_trades']}")
    logger.info("\nTop 5 Results:")
    top_5 = results_df.head(5)[["total_return", "win_rate", "sharpe_ratio", "num_trades"]]
    logger.info(f"\n{top_5.to_string()}")

    return results_df, best_params
