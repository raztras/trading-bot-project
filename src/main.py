"""
Main orchestrator for the trading strategy system

This module coordinates all components of the trading strategy:
- Configuration loading
- Data fetching
- Technical indicator calculation
- ML model training
- Backtesting
- Performance metrics
- Export and visualization

Usage:
    python main.py --profile WINNER
    python main.py --profile CONSERVATIVE --days 360
    python main.py --profile WINNER_OPTIMIZED --threads -1  # use all CPU threads
"""

import argparse
import warnings
import os
import multiprocessing
from logs import logger
from ingestion.config_loader import ConfigLoader
from ingestion.data_fetcher import DataFetcher
from metrics.technical import TechnicalIndicators
from models.ml_model import MLModel
from metrics.backtester import Backtester
from metrics.performance import PerformanceMetrics
from exporters.export_chart import ChartGenerator
from exporters.export_csv import DataExporter
from exporters.export_summary import SummaryExporter
from exporters.export_supabase import SupabaseExporter

warnings.filterwarnings("ignore")


class TradingStrategy:
    """Main trading strategy orchestrator"""

    def __init__(
        self,
        config_path="trading_profiles.yaml",
        profile="WINNER",
        threads: int | None = None,
    ):
        """
        Initialize trading strategy

        Args:
            config_path: Path to config YAML file
            profile: Profile name to load
            threads: Number of CPU threads to use (-1 for all). Applies to XGBoost and math libs.
        """
        self.profile = profile
        self.threads = threads

        # Load configuration
        self.profile_config, self.data_config, self.output_config = (
            ConfigLoader.load_profile(config_path, profile)
        )

        # Optionally override thread/parallel settings
        self._configure_threads()

        # Initialize components
        self.ml_model = MLModel(self.profile_config["ml"], self.output_config, profile)
        self.exporter = DataExporter(self.output_config, profile)
        self.chart_generator = ChartGenerator(
            self.profile_config, self.output_config, profile
        )
        self.summary_exporter = SummaryExporter(self.output_config)

        # Initialize Supabase exporter (optional)
        self.supabase_exporter = None
        try:
            self.supabase_exporter = SupabaseExporter()
            logger.info("Supabase exporter enabled")
        except Exception as e:
            logger.warning(f"Supabase exporter disabled: {e}")

    def run(self, days=None):
        """
        Run complete trading strategy pipeline

        Args:
            days: Number of days of data to fetch (uses config default if None)

        Returns:
            dict: Performance metrics
        """
        logger.info("=" * 100)
        logger.info(f"PRODUCTION STRATEGY: {self.profile}")
        logger.info("=" * 100)

        # 1. Fetch data
        days = days or self.data_config["train_days"]
        df = DataFetcher.fetch_ohlcv(
            self.profile_config["market"]["symbol"],
            self.profile_config["market"]["timeframe"],
            days,
        )

        # 2. Calculate indicators
        df = TechnicalIndicators.calculate_all(df, self.profile_config["indicators"])

        # 3. Split train/test
        train_size = int(len(df) * (1 - self.data_config["test_split"]))
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]

        logger.info(f"Train: {len(df_train)} bars, Test: {len(df_test)} bars")

        # 4. Train ML model (if enabled)
        ml_predictions = None
        if self.profile_config["ml"]["enabled"]:
            self.ml_model.train(df_train)
            ml_predictions = self.ml_model.predict(df_test)

        # 5. Backtest
        backtester = Backtester(self.profile_config, ml_predictions)
        trades_df, signals_df, final_equity = backtester.run(df_test)

        # 6. Calculate metrics
        metrics = PerformanceMetrics.calculate(trades_df, self.profile_config)

        # 7. Print results
        PerformanceMetrics.print_results(metrics, trades_df, self.profile, len(df_test))

        # 8. Export CSV
        self.exporter.export(signals_df, trades_df)

        # 9. Generate HTML chart
        self.chart_generator.generate(signals_df)

        # 10. Export performance summary
        self.summary_exporter.export(
            self.profile, metrics, len(df_test), trades_df, days
        )

        # 11. Export to Supabase (optional)
        if self.supabase_exporter:
            try:
                from datetime import datetime
                run_timestamp = datetime.now()

                # Export trades
                self.supabase_exporter.export_trades(trades_df, self.profile, run_timestamp)

                # Export performance summary
                self.supabase_exporter.export_performance_summary(
                    self.profile, metrics, days, len(df_test), trades_df, run_timestamp
                )
            except Exception as e:
                logger.error(f"Failed to export to Supabase: {e}")

        logger.info("=" * 100)
        logger.info("COMPLETE!")
        logger.info("=" * 100)

        return metrics

    def _configure_threads(self):
        """Configure CPU thread usage for XGBoost and common math libraries.

        Honors self.threads when provided. If -1, uses all available cores.
        """
        if self.threads is None:
            return

        threads = self.threads
        if threads == -1 or threads is None or threads <= 0:
            threads = multiprocessing.cpu_count() or os.cpu_count() or 1

        # Set environment variables for common BLAS backends
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)

        # Ensure ML model (XGBoost) uses desired threads
        try:
            ml_cfg = self.profile_config.get("ml", {})
            if ml_cfg.get("enabled", False):
                mp = ml_cfg.get("model_params", {})
                mp["n_jobs"] = threads  # sklearn alias
                mp["nthread"] = threads  # xgboost native alias
                ml_cfg["model_params"] = mp
                self.profile_config["ml"] = ml_cfg
                logger.info(f"Threads configured: {threads} (XGBoost & math libs)")
        except Exception as e:
            logger.warning(f"Could not apply thread configuration: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Production Trading Strategy")
    parser.add_argument(
        "--profile",
        type=str,
        default="WINNER",
        help="Trading profile (WINNER, CONSERVATIVE, AGGRESSIVE, MODERATE)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="trading_profiles.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Days of historical data (default: from config)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads to use (-1 for all). Applies to ML and math libs.",
    )
    args = parser.parse_args()

    strategy = TradingStrategy(
        config_path=args.config, profile=args.profile, threads=args.threads
    )
    strategy.run(days=args.days)


if __name__ == "__main__":
    main()
