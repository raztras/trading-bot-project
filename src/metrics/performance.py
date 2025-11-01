"""
Performance metrics calculation and reporting
"""
import numpy as np


class PerformanceMetrics:
    """Calculate trading performance metrics"""

    @staticmethod
    def calculate(trades_df, profile_config, start_equity=10000):
        """
        Calculate comprehensive performance metrics

        Args:
            trades_df: DataFrame of completed trades
            profile_config: Profile configuration dictionary
            start_equity: Starting capital

        Returns:
            dict: Dictionary of performance metrics
        """
        if len(trades_df) == 0:
            return {}

        # Basic returns
        gross_return = trades_df["pnl"].sum() / start_equity
        num_trades = len(trades_df)
        transaction_cost = num_trades * (
            profile_config["entry"]["entry_fee"] + profile_config["exit"]["exit_fee"]
        )
        net_return = gross_return - transaction_cost

        # Win/loss analysis
        wins = trades_df[trades_df["pnl"] > 0]
        losses = trades_df[trades_df["pnl"] <= 0]

        win_rate = len(wins) / num_trades if num_trades > 0 else 0
        avg_win = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_pct"].mean() if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins["pnl"].sum()
        gross_loss = abs(losses["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Risk metrics
        returns = trades_df["pnl_pct"].values
        sharpe_ratio = (
            (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        )

        # Drawdown
        equity_curve = trades_df["pnl"].cumsum() + start_equity
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()

        return {
            "gross_return": gross_return,
            "net_return": net_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
        }

    @staticmethod
    def print_results(metrics, trades_df, profile, test_bars):
        """
        Print formatted performance results

        Args:
            metrics: Dictionary of metrics
            trades_df: DataFrame of trades
            profile: Profile name
            test_bars: Number of test bars
        """
        if not metrics:
            print("\nNo trades executed!")
            return

        print("\n" + "=" * 100)
        print(f"RESULTS: {profile}")
        print("=" * 100)

        print(f"\n[PERFORMANCE]")
        print(f"  NET Return:        {metrics['net_return'] * 100:+.2f}%")
        print(
            f"  Annualized:        {(metrics['net_return'] / (test_bars / 24)) * 365 * 100:.1f}%"
        )
        print(f"  Trades:            {metrics['num_trades']}")
        print(f"  Win Rate:          {metrics['win_rate'] * 100:.1f}%")
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:      {metrics['max_drawdown'] * 100:.2f}%")

        print(f"\n[RISK/REWARD]")
        print(f"  Avg Win:           {metrics['avg_win'] * 100:+.2f}%")
        print(f"  Avg Loss:          {metrics['avg_loss'] * 100:+.2f}%")
        print(
            f"  Win/Loss Ratio:    {abs(metrics['avg_win'] / metrics['avg_loss']):.2f}:1"
            if metrics["avg_loss"] != 0
            else "  Win/Loss Ratio:    N/A"
        )

        print(f"\n[EXIT REASONS]")
        if len(trades_df) > 0:
            for reason, count in trades_df["exit_reason"].value_counts().items():
                print(
                    f"  {reason:20s}: {count:3d} ({count / len(trades_df) * 100:.1f}%)"
                )
