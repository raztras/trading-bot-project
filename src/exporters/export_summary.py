"""
Export summary statistics to CSV
"""
import os
import pandas as pd
from datetime import datetime
from logs import logger


class SummaryExporter:
    """Export performance summary statistics to CSV"""

    def __init__(self, output_config):
        """
        Initialize summary exporter

        Args:
            output_config: Output configuration dictionary
        """
        self.output_config = output_config

    def export(self, profile, metrics, test_bars, trades_df, days_tested):
        """
        Export summary statistics to CSV

        Args:
            profile: Profile name
            metrics: Dictionary of performance metrics
            test_bars: Number of test bars
            trades_df: DataFrame of trades
            days_tested: Total days of data tested

        Returns:
            str: Path to summary CSV file
        """
        if not metrics:
            logger.warning("No metrics to export")
            return None

        output_dir = self.output_config["base_path"]
        os.makedirs(output_dir, exist_ok=True)

        # Calculate additional metrics
        test_days = test_bars / 24  # Convert bars to days
        annualized_return = (metrics["net_return"] / test_days) * 365 * 100
        trades_per_year = (metrics["num_trades"] / test_days) * 365

        # Build summary dict
        summary = {
            "profile": profile,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "days_tested": days_tested,
            "test_bars": test_bars,
            "test_days": round(test_days, 1),
            "net_return_pct": round(metrics["net_return"] * 100, 2),
            "gross_return_pct": round(metrics["gross_return"] * 100, 2),
            "annualized_return_pct": round(annualized_return, 1),
            "total_trades": metrics["num_trades"],
            "trades_per_year": round(trades_per_year, 1),
            "win_rate_pct": round(metrics["win_rate"] * 100, 1),
            "profit_factor": round(metrics["profit_factor"], 2),
            "sharpe_ratio": round(metrics["sharpe_ratio"], 2),
            "max_drawdown_pct": round(metrics["max_drawdown"] * 100, 2),
            "avg_win_pct": round(metrics["avg_win"] * 100, 2),
            "avg_loss_pct": round(metrics["avg_loss"] * 100, 2),
        }

        # Add win/loss ratio
        if metrics["avg_loss"] != 0:
            summary["win_loss_ratio"] = round(
                abs(metrics["avg_win"] / metrics["avg_loss"]), 2
            )
        else:
            summary["win_loss_ratio"] = None

        # Add exit reason breakdown
        if len(trades_df) > 0:
            exit_counts = trades_df["exit_reason"].value_counts()
            for reason, count in exit_counts.items():
                summary[f"exit_{reason}"] = count
                summary[f"exit_{reason}_pct"] = round(
                    count / len(trades_df) * 100, 1
                )

        # Convert to DataFrame and append to CSV
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(output_dir, "performance_summary.csv")

        # Append if file exists, otherwise create new
        if os.path.exists(summary_path):
            existing_df = pd.read_csv(summary_path)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)

        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary exported to {summary_path}")

        return summary_path
