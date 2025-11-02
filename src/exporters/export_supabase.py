"""
Export trading data to Supabase for analytics
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from logs import logger
import pandas as pd


class SupabaseExporter:
    """Export trades and performance metrics to Supabase"""

    def __init__(self):
        """
        Initialize Supabase client with credentials from .env

        Required .env variables:
            - sb_db: Supabase database URL
            - sb_api: Supabase API key
        """
        # Load environment variables from root .env file
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(env_path)

        self.url = os.getenv("sb_db")
        self.key = os.getenv("sb_api")

        if not self.url or not self.key:
            raise ValueError("Missing Supabase credentials. Check .env file for sb_db and sb_api")

        self.client: Client = create_client(self.url, self.key)
        logger.info("Supabase client initialized")

    def export_trades(self, trades_df: pd.DataFrame, profile: str, run_timestamp: datetime = None):
        """
        Export trades to Supabase trades table

        Args:
            trades_df: DataFrame with trade data (from backtester)
            profile: Trading profile name
            run_timestamp: Timestamp of this backtest run

        Returns:
            int: Number of rows inserted
        """
        if trades_df is None or len(trades_df) == 0:
            logger.warning("No trades to export to Supabase")
            return 0

        run_timestamp = run_timestamp or datetime.now()

        # Prepare data for insertion
        trades_data = []
        for _, row in trades_df.iterrows():
            trade = {
                "profile": profile,
                "run_timestamp": run_timestamp.isoformat(),
                "entry_time": pd.Timestamp(row["entry_time"]).isoformat(),
                "exit_time": pd.Timestamp(row["exit_time"]).isoformat(),
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "pnl": float(row["pnl"]),
                "pnl_pct": float(row["pnl_pct"]),
                "bars_held": int(row["bars_held"]),
                "exit_reason": row["exit_reason"],
                "ml_predicted_gain": float(row.get("ml_predicted_gain", 0.0)) if pd.notna(row.get("ml_predicted_gain")) else None,
            }
            trades_data.append(trade)

        try:
            # Insert in batches of 1000
            batch_size = 1000
            total_inserted = 0

            for i in range(0, len(trades_data), batch_size):
                batch = trades_data[i:i + batch_size]
                response = self.client.table("trades").insert(batch).execute()
                total_inserted += len(batch)

            logger.info(f"✅ Exported {total_inserted} trades to Supabase")
            return total_inserted

        except Exception as e:
            logger.error(f"❌ Failed to export trades to Supabase: {e}")
            raise

    def export_performance_summary(
        self,
        profile: str,
        metrics: dict,
        days_tested: int,
        test_bars: int,
        trades_df: pd.DataFrame,
        run_timestamp: datetime = None
    ):
        """
        Export performance summary to Supabase

        Args:
            profile: Trading profile name
            metrics: Performance metrics dictionary
            days_tested: Number of days in backtest
            test_bars: Number of bars tested
            trades_df: DataFrame with trade data (for exit reason stats)
            run_timestamp: Timestamp of this backtest run

        Returns:
            dict: Inserted row data
        """
        run_timestamp = run_timestamp or datetime.now()

        # Build summary data (exit reasons can be calculated from trades table)
        summary_data = {
            "profile": profile,
            "run_timestamp": run_timestamp.isoformat(),
            "days_tested": days_tested,
            "test_bars": test_bars,
            "net_return_pct": float(metrics.get("net_return_pct", 0)),
            "annualized_return_pct": float(metrics.get("annualized_return", 0)),
            "total_trades": int(metrics.get("total_trades", 0)),
            "win_rate_pct": float(metrics.get("win_rate", 0)),
            "sharpe_ratio": float(metrics.get("sharpe_ratio", 0)) if metrics.get("sharpe_ratio") else None,
            "max_drawdown_pct": float(metrics.get("max_drawdown", 0)),
            "profit_factor": float(metrics.get("profit_factor", 0)) if metrics.get("profit_factor") else None,
        }

        try:
            response = self.client.table("performance_summary").insert(summary_data).execute()
            logger.info(f"✅ Exported performance summary to Supabase")
            return summary_data

        except Exception as e:
            logger.error(f"❌ Failed to export performance summary to Supabase: {e}")
            raise

    def upsert_current_signal(
        self,
        symbol: str,
        profile: str,
        signal_data: dict
    ):
        """
        Upsert current trading signal (for live trading monitoring)

        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            profile: Trading profile name
            signal_data: Dictionary with current signal data

        Returns:
            dict: Upserted row data
        """
        current_signal = {
            "symbol": symbol,
            "profile": profile,
            "timestamp": signal_data.get("timestamp", datetime.now()).isoformat(),
            "close_price": float(signal_data.get("close_price", 0)),
            "sma_fast": float(signal_data.get("sma_fast", 0)),
            "sma_slow": float(signal_data.get("sma_slow", 0)),
            "rsi": float(signal_data.get("rsi", 0)) if signal_data.get("rsi") else None,
            "in_position": bool(signal_data.get("in_position", False)),
            "ml_prediction": float(signal_data.get("ml_prediction", 0)) if signal_data.get("ml_prediction") else None,
            "updated_at": datetime.now().isoformat()
        }

        try:
            response = self.client.table("current_signals").upsert(current_signal).execute()
            logger.info(f"✅ Updated current signal for {symbol} in Supabase")
            return current_signal

        except Exception as e:
            logger.error(f"❌ Failed to upsert current signal to Supabase: {e}")
            raise

    def test_connection(self):
        """
        Test Supabase connection by checking if tables exist

        Returns:
            bool: True if connection successful
        """
        try:
            # Try to query trades table (will fail if doesn't exist)
            response = self.client.table("trades").select("id").limit(1).execute()
            logger.info("✅ Supabase connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            logger.info("Make sure you've created the required tables in Supabase")
            return False
