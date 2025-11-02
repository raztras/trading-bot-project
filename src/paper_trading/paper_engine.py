"""
Paper trading engine - executes live trading strategy
"""
from datetime import datetime, timedelta
import pandas as pd
import pickle
import os
from logs import logger
from paper_trading.paper_db import PaperTradingDB
from ingestion.data_fetcher import DataFetcher
from metrics.technical import TechnicalIndicators


class PaperTradingEngine:
    """Executes paper trading strategy based on config"""

    def __init__(self, profile_config, profile_name: str):
        """
        Initialize paper trading engine

        Args:
            profile_config: Trading profile configuration
            profile_name: Profile name (e.g., 'OPTIMIZED')
        """
        self.profile_config = profile_config
        self.profile_name = profile_name
        self.db = PaperTradingDB()

        # Extract config
        self.symbol = profile_config["market"]["symbol"]
        self.timeframe = profile_config["market"]["timeframe"]
        self.indicators_config = profile_config["indicators"]
        self.entry_config = profile_config["entry"]
        self.exit_config = profile_config["exit"]
        self.ml_config = profile_config["ml"]

        # Load ML model if enabled
        self.ml_model = None
        self.ml_features = None
        if self.ml_config["enabled"]:
            self._load_ml_model()

        logger.info(f"Paper Trading Engine initialized: {profile_name}")

    def _load_ml_model(self):
        """Load trained ML model from disk"""
        try:
            model_path = os.path.join("output", f"ml_model_{self.profile_name}.pkl")

            # Try relative path
            if not os.path.exists(model_path):
                # Try from src directory
                model_path = os.path.join("..", "output", f"ml_model_{self.profile_name}.pkl")

            if not os.path.exists(model_path):
                logger.warning(f"ML model not found: {model_path}")
                logger.warning("Run a backtest first to train the model")
                return

            with open(model_path, "rb") as f:
                model_data = pickle.load(f)
                self.ml_model = model_data["model"]
                self.ml_features = model_data["features"]

            logger.info(f"✅ Loaded ML model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.ml_model = None

    def run(self):
        """
        Main execution loop - runs once per scheduled interval

        Steps:
        1. Fetch latest market data
        2. Calculate indicators
        3. Check for signals
        4. Manage positions (entry/exit)
        5. Update portfolio state
        6. Log everything to Supabase
        """
        logger.info("=" * 80)
        logger.info(f"PAPER TRADING: {self.profile_name} | {datetime.now()}")
        logger.info("=" * 80)

        try:
            # 1. Get or create session
            session = self._get_or_create_session()

            # 2. Fetch latest data (need enough for indicators)
            df = self._fetch_market_data()

            # 3. Calculate indicators
            df = TechnicalIndicators.calculate_all(df, self.indicators_config)

            # 4. Get latest candle
            latest = df.iloc[-1]
            current_price = float(latest["close"])
            timestamp = pd.Timestamp(latest.name)

            logger.info(f"Current price: ${current_price:.2f}")
            logger.info(f"SMA Fast: ${latest['sma_fast']:.2f} | SMA Slow: ${latest['sma_slow']:.2f}")
            logger.info(f"RSI: {latest['rsi']:.2f}")

            # 5. Get ML prediction if enabled
            ml_prediction = None
            if self.ml_model and self.ml_features:
                try:
                    features = df[self.ml_features].iloc[-1:].fillna(0).values
                    ml_prediction = float(self.ml_model.predict(features)[0])
                    logger.info(f"ML Prediction: {ml_prediction:.4f} ({ml_prediction*100:.2f}%)")
                except Exception as e:
                    logger.error(f"ML prediction failed: {e}")

            # 6. Check current position
            open_position = self.db.get_open_position(self.profile_name, self.symbol)
            portfolio = self.db.get_latest_portfolio(self.profile_name)

            # 7. Make trading decision
            if open_position:
                self._handle_open_position(open_position, latest, timestamp, ml_prediction, portfolio, session)
            else:
                self._handle_no_position(latest, timestamp, ml_prediction, portfolio, session)

            # 8. Update portfolio snapshot
            self._save_portfolio_snapshot(timestamp, current_price, portfolio, session)

            logger.info("=" * 80)
            logger.info("Paper trading cycle complete")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Paper trading error: {e}", exc_info=True)
            raise

    def _get_or_create_session(self):
        """Get active session or create new one"""
        session = self.db.get_active_session(self.profile_name)

        if not session:
            logger.info("No active session found, creating new one...")
            session = self.db.create_session(self.profile_name, initial_cash=100000.0)

            # Initialize portfolio
            portfolio_data = {
                "timestamp": datetime.now(),
                "cash_balance": 100000.0,
                "position_value": 0.0,
                "total_equity": 100000.0,
                "in_position": False,
                "total_trades": 0,
                "winning_trades": 0,
                "total_pnl": 0.0,
            }
            self.db.save_portfolio_snapshot(self.profile_name, portfolio_data)
        else:
            logger.info(f"Using active session: {session['session_name']}")

        return session

    def _fetch_market_data(self):
        """Fetch recent market data for analysis"""
        # Fetch enough data for indicators (max lookback + buffer)
        days = 30  # Enough for SMAs and indicators

        logger.info(f"Fetching {days} days of {self.timeframe} data for {self.symbol}")
        df = DataFetcher.fetch_ohlcv(self.symbol, self.timeframe, days)
        logger.info(f"Fetched {len(df)} candles")

        return df

    def _handle_no_position(self, latest, timestamp, ml_prediction, portfolio, session):
        """Handle logic when no position is open"""
        signal_type = "hold"
        action_taken = "none"
        reason = "No signal"

        # Check for buy signal (SMA cross up)
        if latest["sma_cross_up"]:
            # Apply ML filter if enabled
            if self.ml_config["enabled"] and ml_prediction is not None:
                ml_threshold = self.ml_config.get("threshold", 0.01)

                if ml_prediction >= ml_threshold:
                    signal_type = "buy"
                    action_taken = "entered"
                    reason = f"SMA cross up + ML prediction {ml_prediction:.4f} > {ml_threshold}"

                    # Enter position
                    self._enter_position(latest, timestamp, ml_prediction, portfolio)
                else:
                    signal_type = "buy"
                    action_taken = "ignored"
                    reason = f"SMA cross up but ML prediction {ml_prediction:.4f} < {ml_threshold}"
            else:
                # No ML, enter on SMA cross up
                signal_type = "buy"
                action_taken = "entered"
                reason = "SMA cross up"

                self._enter_position(latest, timestamp, ml_prediction, portfolio)

        # Log signal
        signal_data = {
            "timestamp": timestamp,
            "signal_type": signal_type,
            "close_price": latest["close"],
            "sma_fast": latest["sma_fast"],
            "sma_slow": latest["sma_slow"],
            "rsi": latest["rsi"],
            "ml_prediction": ml_prediction,
            "action_taken": action_taken,
            "reason": reason,
        }
        self.db.log_signal(self.profile_name, self.symbol, signal_data)

        if action_taken == "entered":
            logger.info(f"🟢 ENTERED POSITION: {reason}")
        else:
            logger.info(f"⚪ NO ACTION: {reason}")

    def _handle_open_position(self, position, latest, timestamp, ml_prediction, portfolio, session):
        """Handle logic when position is open"""
        signal_type = "hold"
        action_taken = "none"
        reason = "Holding position"

        entry_price = float(position["entry_price"])
        current_price = float(latest["close"])
        entry_time = pd.Timestamp(position["entry_time"])
        bars_held = int((timestamp - entry_time).total_seconds() / 3600)  # Assuming 1h bars

        # Calculate unrealized P&L
        position_size = float(position["position_size"])
        unrealized_pnl_pct = (current_price - entry_price) / entry_price

        logger.info(f"📊 Open position: Entry ${entry_price:.2f} | Current ${current_price:.2f} | "
                   f"Unrealized P&L: {unrealized_pnl_pct*100:.2f}% | Bars held: {bars_held}")

        # Check exit conditions
        should_exit = False
        exit_reason = None

        # 1. Profit target
        profit_target = self.exit_config["profit_target"]
        if unrealized_pnl_pct >= profit_target:
            should_exit = True
            exit_reason = "profit_target"
            reason = f"Profit target reached: {unrealized_pnl_pct*100:.2f}% >= {profit_target*100:.2f}%"

        # 2. Stop loss
        if not should_exit:
            stop_loss = self.exit_config["stop_loss"]
            if unrealized_pnl_pct <= -stop_loss:
                should_exit = True
                exit_reason = "stop_loss"
                reason = f"Stop loss hit: {unrealized_pnl_pct*100:.2f}% <= -{stop_loss*100:.2f}%"

        # 3. SMA cross down
        if not should_exit and self.exit_config.get("exit_on_sma_cross_down", True):
            if latest["sma_cross_down"]:
                should_exit = True
                exit_reason = "sma_cross_down"
                reason = "SMA cross down signal"

        # 4. Max hold period
        if not should_exit:
            max_hold_hours = self.exit_config["max_hold_hours"]
            if bars_held >= max_hold_hours:
                should_exit = True
                exit_reason = "max_hold"
                reason = f"Max hold period reached: {bars_held} >= {max_hold_hours} hours"

        # Execute exit if needed
        if should_exit:
            signal_type = "sell"
            action_taken = "exited"
            self._exit_position(position, latest, timestamp, exit_reason, bars_held, portfolio, session)
            logger.info(f"🔴 EXITED POSITION: {reason}")
        else:
            logger.info(f"⚪ HOLDING: {reason}")

        # Log signal
        signal_data = {
            "timestamp": timestamp,
            "signal_type": signal_type,
            "close_price": latest["close"],
            "sma_fast": latest["sma_fast"],
            "sma_slow": latest["sma_slow"],
            "rsi": latest["rsi"],
            "ml_prediction": ml_prediction,
            "action_taken": action_taken,
            "reason": reason,
        }
        self.db.log_signal(self.profile_name, self.symbol, signal_data)

    def _enter_position(self, latest, timestamp, ml_prediction, portfolio):
        """Enter a new position"""
        current_cash = float(portfolio["cash_balance"]) if portfolio else 100000.0
        entry_price = float(latest["close"])

        # Use all available cash (100% allocation)
        entry_cash = current_cash
        position_size = entry_cash / entry_price

        entry_data = {
            "entry_time": timestamp,
            "entry_price": entry_price,
            "position_size": position_size,
            "entry_cash": entry_cash,
            "ml_predicted_gain": ml_prediction,
        }

        self.db.open_position(self.profile_name, self.symbol, entry_data)

    def _exit_position(self, position, latest, timestamp, exit_reason, bars_held, portfolio, session):
        """Exit an existing position"""
        entry_price = float(position["entry_price"])
        exit_price = float(latest["close"])
        position_size = float(position["position_size"])
        entry_cash = float(position["entry_cash"])

        # Calculate P&L
        exit_cash = position_size * exit_price
        pnl = exit_cash - entry_cash
        pnl_pct = (exit_price - entry_price) / entry_price * 100

        exit_data = {
            "exit_time": timestamp,
            "exit_price": exit_price,
            "exit_cash": exit_cash,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
        }

        self.db.close_position(position["id"], exit_data)

        # Update session metrics
        all_closed = self.db.get_all_positions(self.profile_name, status="closed")
        total_trades = len(all_closed)
        winning_trades = sum(1 for p in all_closed if float(p["pnl"]) > 0)
        total_pnl = sum(float(p["pnl"]) for p in all_closed)

        metrics = {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "win_rate_pct": (winning_trades / total_trades * 100) if total_trades > 0 else 0,
        }
        self.db.update_session_metrics(session["id"], metrics)

    def _save_portfolio_snapshot(self, timestamp, current_price, portfolio, session):
        """Save current portfolio state"""
        # Get open position
        open_position = self.db.get_open_position(self.profile_name, self.symbol)

        # Get all closed positions for stats
        all_closed = self.db.get_all_positions(self.profile_name, status="closed")
        total_trades = len(all_closed)
        winning_trades = sum(1 for p in all_closed if float(p["pnl"]) > 0)
        total_pnl = sum(float(p["pnl"]) for p in all_closed)

        if open_position:
            # In position
            entry_cash = float(open_position["entry_cash"])
            position_size = float(open_position["position_size"])
            position_value = position_size * current_price
            unrealized_pnl = position_value - entry_cash

            portfolio_data = {
                "timestamp": timestamp,
                "cash_balance": 0.0,
                "position_value": position_value,
                "total_equity": position_value,
                "in_position": True,
                "current_symbol": self.symbol,
                "position_entry_price": float(open_position["entry_price"]),
                "position_size": position_size,
                "unrealized_pnl": unrealized_pnl,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "total_pnl": total_pnl,
            }
        else:
            # Not in position - cash only
            if portfolio:
                cash_balance = float(portfolio["cash_balance"])
            else:
                # First time, use initial cash + realized PnL
                cash_balance = session["initial_cash"] + total_pnl

            portfolio_data = {
                "timestamp": timestamp,
                "cash_balance": cash_balance,
                "position_value": 0.0,
                "total_equity": cash_balance,
                "in_position": False,
                "current_symbol": None,
                "position_entry_price": None,
                "position_size": None,
                "unrealized_pnl": None,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "total_pnl": total_pnl,
            }

        self.db.save_portfolio_snapshot(self.profile_name, portfolio_data)

        logger.info(f"💰 Portfolio: ${portfolio_data['total_equity']:.2f} | "
                   f"Trades: {total_trades} ({winning_trades}W) | "
                   f"Total P&L: ${total_pnl:.2f}")
