"""
Supabase data access layer for paper trading
"""
import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from logs import logger
import pandas as pd


class PaperTradingDB:
    """Manages all paper trading data in Supabase"""

    def __init__(self):
        """Initialize Supabase client"""
        # Load environment variables
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(env_path)

        self.url = os.getenv("sb_db")
        self.key = os.getenv("sb_api")

        if not self.url or not self.key:
            raise ValueError("Missing Supabase credentials in .env")

        self.client: Client = create_client(self.url, self.key)

    # ==================== SESSIONS ====================

    def get_active_session(self, profile: str):
        """Get active paper trading session for profile"""
        try:
            response = (
                self.client.table("paper_sessions")
                .select("*")
                .eq("profile", profile)
                .eq("status", "active")
                .order("started_at", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get active session: {e}")
            return None

    def create_session(self, profile: str, initial_cash: float = 100000.0):
        """Create new paper trading session"""
        try:
            session_data = {
                "profile": profile,
                "session_name": f"{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "started_at": datetime.now().isoformat(),
                "status": "active",
                "initial_cash": initial_cash,
            }
            response = self.client.table("paper_sessions").insert(session_data).execute()
            logger.info(f"✅ Created paper trading session: {session_data['session_name']}")
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise

    def update_session_metrics(self, session_id: int, metrics: dict):
        """Update session final metrics"""
        try:
            self.client.table("paper_sessions").update(metrics).eq("id", session_id).execute()
        except Exception as e:
            logger.error(f"Failed to update session metrics: {e}")

    def end_session(self, session_id: int, final_metrics: dict):
        """End paper trading session"""
        try:
            update_data = {
                "ended_at": datetime.now().isoformat(),
                "status": "stopped",
                **final_metrics,
            }
            self.client.table("paper_sessions").update(update_data).eq("id", session_id).execute()
            logger.info(f"✅ Ended paper trading session {session_id}")
        except Exception as e:
            logger.error(f"Failed to end session: {e}")

    # ==================== POSITIONS ====================

    def get_open_position(self, profile: str, symbol: str):
        """Get current open position for profile/symbol"""
        try:
            response = (
                self.client.table("paper_positions")
                .select("*")
                .eq("profile", profile)
                .eq("symbol", symbol)
                .eq("status", "open")
                .order("entry_time", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get open position: {e}")
            return None

    def open_position(self, profile: str, symbol: str, entry_data: dict):
        """Open new position"""
        try:
            position_data = {
                "profile": profile,
                "symbol": symbol,
                "entry_time": entry_data["entry_time"].isoformat(),
                "entry_price": float(entry_data["entry_price"]),
                "position_size": float(entry_data["position_size"]),
                "entry_cash": float(entry_data["entry_cash"]),
                "status": "open",
                "ml_predicted_gain": float(entry_data.get("ml_predicted_gain", 0)) if entry_data.get("ml_predicted_gain") else None,
            }
            response = self.client.table("paper_positions").insert(position_data).execute()
            logger.info(f"✅ Opened position: {symbol} @ ${entry_data['entry_price']:.2f}")
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            raise

    def close_position(self, position_id: int, exit_data: dict):
        """Close existing position"""
        try:
            update_data = {
                "exit_time": exit_data["exit_time"].isoformat(),
                "exit_price": float(exit_data["exit_price"]),
                "exit_cash": float(exit_data["exit_cash"]),
                "pnl": float(exit_data["pnl"]),
                "pnl_pct": float(exit_data["pnl_pct"]),
                "status": "closed",
                "exit_reason": exit_data["exit_reason"],
                "bars_held": int(exit_data.get("bars_held", 0)),
                "updated_at": datetime.now().isoformat(),
            }
            response = (
                self.client.table("paper_positions")
                .update(update_data)
                .eq("id", position_id)
                .execute()
            )
            logger.info(
                f"✅ Closed position: {exit_data['exit_reason']} | "
                f"PnL: ${exit_data['pnl']:.2f} ({exit_data['pnl_pct']:.2f}%)"
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            raise

    def get_all_positions(self, profile: str, status: str = None):
        """Get all positions for profile"""
        try:
            query = self.client.table("paper_positions").select("*").eq("profile", profile)
            if status:
                query = query.eq("status", status)
            response = query.order("entry_time", desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return []

    # ==================== PORTFOLIO ====================

    def get_latest_portfolio(self, profile: str):
        """Get latest portfolio state"""
        try:
            response = (
                self.client.table("paper_portfolio")
                .select("*")
                .eq("profile", profile)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Failed to get portfolio: {e}")
            return None

    def save_portfolio_snapshot(self, profile: str, portfolio_data: dict):
        """Save portfolio snapshot"""
        try:
            snapshot = {
                "profile": profile,
                "timestamp": portfolio_data["timestamp"].isoformat(),
                "cash_balance": float(portfolio_data["cash_balance"]),
                "position_value": float(portfolio_data["position_value"]),
                "total_equity": float(portfolio_data["total_equity"]),
                "in_position": bool(portfolio_data["in_position"]),
                "current_symbol": portfolio_data.get("current_symbol"),
                "position_entry_price": float(portfolio_data["position_entry_price"]) if portfolio_data.get("position_entry_price") else None,
                "position_size": float(portfolio_data["position_size"]) if portfolio_data.get("position_size") else None,
                "unrealized_pnl": float(portfolio_data["unrealized_pnl"]) if portfolio_data.get("unrealized_pnl") else None,
                "total_trades": int(portfolio_data["total_trades"]),
                "winning_trades": int(portfolio_data["winning_trades"]),
                "total_pnl": float(portfolio_data["total_pnl"]),
            }
            response = self.client.table("paper_portfolio").insert(snapshot).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to save portfolio snapshot: {e}")
            raise

    def get_portfolio_history(self, profile: str, limit: int = 100):
        """Get portfolio history for equity curve"""
        try:
            response = (
                self.client.table("paper_portfolio")
                .select("timestamp,total_equity")
                .eq("profile", profile)
                .order("timestamp", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Failed to get portfolio history: {e}")
            return []

    # ==================== SIGNALS ====================

    def log_signal(self, profile: str, symbol: str, signal_data: dict):
        """Log a trading signal"""
        try:
            log_data = {
                "profile": profile,
                "symbol": symbol,
                "timestamp": signal_data["timestamp"].isoformat(),
                "signal_type": signal_data["signal_type"],
                "close_price": float(signal_data["close_price"]),
                "sma_fast": float(signal_data.get("sma_fast", 0)) if signal_data.get("sma_fast") else None,
                "sma_slow": float(signal_data.get("sma_slow", 0)) if signal_data.get("sma_slow") else None,
                "rsi": float(signal_data.get("rsi", 0)) if signal_data.get("rsi") else None,
                "ml_prediction": float(signal_data.get("ml_prediction", 0)) if signal_data.get("ml_prediction") else None,
                "action_taken": signal_data.get("action_taken"),
                "reason": signal_data.get("reason"),
            }
            response = self.client.table("paper_signals").insert(log_data).execute()
            return response.data[0]
        except Exception as e:
            logger.error(f"Failed to log signal: {e}")
            return None
