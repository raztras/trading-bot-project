#!/usr/bin/env python3
"""
Paper Trading Runner - Execute live paper trading strategy

Run this script hourly via cron:
5 * * * * cd /path/to/trading-bot && /path/to/python run_paper_trading.py --profile OPTIMIZED

This runs at :05 past each hour to ensure clean hourly OHLC data
"""
import sys
import os
import argparse
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from logs import logger
from ingestion.config_loader import ConfigLoader
from paper_trading.paper_engine import PaperTradingEngine

warnings.filterwarnings("ignore")


def main():
    """Main entry point for paper trading"""
    parser = argparse.ArgumentParser(description="Paper Trading Strategy")
    parser.add_argument(
        "--profile",
        type=str,
        default="OPTIMIZED",
        help="Trading profile to use (OPTIMIZED, WINNER, etc.)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/trading_profiles.yaml",
        help="Path to config file",
    )
    args = parser.parse_args()

    try:
        # Load configuration
        logger.info(f"Loading profile: {args.profile}")
        profile_config, data_config, output_config = ConfigLoader.load_profile(
            args.config, args.profile
        )

        # Initialize and run paper trading
        engine = PaperTradingEngine(profile_config, args.profile)
        engine.run()

        logger.info("✅ Paper trading cycle completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.info("\n⚠️  Paper trading interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Paper trading failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
