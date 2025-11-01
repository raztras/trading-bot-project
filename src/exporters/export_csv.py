"""
Export utilities for signals and trades
"""
import os
from datetime import datetime
from logs import logger


class DataExporter:
    """Export trading data to CSV files"""

    def __init__(self, output_config, profile):
        """
        Initialize data exporter

        Args:
            output_config: Output configuration dictionary
            profile: Profile name
        """
        self.output_config = output_config
        self.profile = profile

    def export(self, signals_df, trades_df):
        """
        Export signals and trades to CSV

        Args:
            signals_df: DataFrame with signals
            trades_df: DataFrame with trades

        Returns:
            tuple: (signals_path, trades_path)
        """
        output_dir = self.output_config["base_path"]
        os.makedirs(output_dir, exist_ok=True)

        signals_path = None
        trades_path = None

        # Export signals
        if self.output_config.get("csv_signals", True) and len(signals_df) > 0:
            signals_path = os.path.join(
                output_dir, "signals.csv"
            )
            signals_df.to_csv(signals_path, index=False)
            logger.info(f"Signals exported to {signals_path}")

        # Export trades
        if self.output_config.get("csv_trades", True) and len(trades_df) > 0:
            trades_path = os.path.join(
                output_dir, "trades.csv"
            )
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Trades exported to {trades_path}")

        return signals_path, trades_path
