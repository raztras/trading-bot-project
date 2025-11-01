"""
Backtesting engine for trading strategies
"""
import pandas as pd
from logs import logger


class Backtester:
    """Backtest trading strategies with entry/exit rules"""

    def __init__(self, profile_config, ml_predictions=None):
        """
        Initialize backtester

        Args:
            profile_config: Profile configuration dictionary
            ml_predictions: Array of ML predictions (optional)
        """
        self.profile_config = profile_config
        self.ml_predictions = ml_predictions
        self.entry_cfg = profile_config["entry"]
        self.exit_cfg = profile_config["exit"]
        self.ml_cfg = profile_config["ml"]

    def run(self, df, start_equity=10000):
        """
        Run backtest on data

        Args:
            df: DataFrame with indicators
            start_equity: Starting capital

        Returns:
            tuple: (trades_df, signals_df, final_equity)
        """
        logger.info("Running backtest...")

        cash = start_equity
        position = 0
        trades = []
        signals = []
        entry_price = 0
        entry_idx = None
        bars_held = 0

        # Backtest loop
        for i, (idx, row) in enumerate(df.iterrows()):
            current_price = row["close"]
            signal = "HOLD"

            # EXIT logic
            if position > 0:
                bars_held += 1
                pnl_pct = (current_price - entry_price) / entry_price
                exit_reason = None

                # Check exit conditions
                if current_price <= entry_price * (1 - self.exit_cfg["stop_loss"]):
                    exit_reason = "stop_loss"
                    signal = "SELL"
                elif current_price >= entry_price * (1 + self.exit_cfg["profit_target"]):
                    exit_reason = "profit_target"
                    signal = "SELL"
                elif (
                    self.exit_cfg.get("exit_on_sma_cross_down", True)
                    and row["sma_cross_down"]
                ):
                    exit_reason = "sma_cross_down"
                    signal = "SELL"
                elif bars_held >= self.exit_cfg["max_hold_hours"]:
                    exit_reason = "max_hold"
                    signal = "SELL"

                # Execute exit
                if exit_reason:
                    exit_price = current_price
                    pnl = position * (exit_price - entry_price)
                    cash += position * exit_price * (1 - self.exit_cfg["exit_fee"])

                    trade_data = {
                        "entry_time": entry_idx,
                        "exit_time": idx,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "bars_held": bars_held,
                        "exit_reason": exit_reason,
                    }

                    if self.ml_predictions is not None:
                        trade_data["ml_predicted_gain"] = self.ml_predictions[i]

                    trades.append(trade_data)

                    position = 0
                    bars_held = 0

            # ENTRY logic
            if position == 0 and row["sma_cross_up"]:
                # Check all entry conditions
                volume_ok = (
                    row["high_volume"]
                    if self.entry_cfg.get("require_volume", False)
                    else True
                )
                ml_ok = True
                if (
                    self.entry_cfg.get("require_ml", False)
                    and self.ml_predictions is not None
                ):
                    ml_ok = self.ml_predictions[i] > self.ml_cfg["threshold"]

                # Execute entry
                if volume_ok and ml_ok:
                    position = (cash * (1 - self.entry_cfg["entry_fee"])) / current_price
                    cash = 0
                    entry_price = current_price
                    entry_idx = idx
                    bars_held = 0
                    signal = "BUY"

            # Record signal
            signal_data = {
                "timestamp": idx,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "sma_fast": row["sma_fast"],
                "sma_slow": row["sma_slow"],
                "volume_ma": row["volume_ma"],
                "rsi": row["rsi"],
                "bb_upper": row["bb_upper"],
                "bb_middle": row["bb_middle"],
                "bb_lower": row["bb_lower"],
                "signal": signal,
            }

            if self.ml_predictions is not None:
                signal_data["ml_prediction"] = self.ml_predictions[i]

            if position > 0:
                signal_data["position_pnl_pct"] = (
                    current_price - entry_price
                ) / entry_price

            signals.append(signal_data)

        trades_df = pd.DataFrame(trades)
        signals_df = pd.DataFrame(signals)

        final_equity = cash + (position * df.iloc[-1]["close"] if position > 0 else 0)

        logger.info(
            f"Backtest complete: {len(trades)} trades, Final equity: ${final_equity:.2f}"
        )

        return trades_df, signals_df, final_equity
