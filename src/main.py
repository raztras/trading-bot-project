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
"""
import argparse
import warnings
from logs import logger
from ingestion.config_loader import ConfigLoader
from ingestion.data_fetcher import DataFetcher
from metrics.technical import TechnicalIndicators
from models.ml_model import MLModel
from metrics.backtester import Backtester
from metrics.performance import PerformanceMetrics
from exporters.export_chart import ChartGenerator
from exporters.export_csv import DataExporter

warnings.filterwarnings("ignore")


class TradingStrategy:
    """Main trading strategy orchestrator"""

    def __init__(self, config_path="trading_profiles.yaml", profile="WINNER"):
        """
        Initialize trading strategy

        Args:
            config_path: Path to config YAML file
            profile: Profile name to load
        """
        self.profile = profile

        # Load configuration
        self.profile_config, self.data_config, self.output_config = ConfigLoader.load_profile(
            config_path, profile
        )

        # Initialize components
        self.ml_model = MLModel(
            self.profile_config["ml"], self.output_config, profile
        )
        self.exporter = DataExporter(self.output_config, profile)
        self.chart_generator = ChartGenerator(
            self.profile_config, self.output_config, profile
        )

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

        logger.info("=" * 100)
        logger.info("COMPLETE!")
        logger.info("=" * 100)

        return metrics


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
    args = parser.parse_args()

    strategy = TradingStrategy(config_path=args.config, profile=args.profile)
    strategy.run(days=args.days)


if __name__ == "__main__":
    main()
