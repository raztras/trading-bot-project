"""
HTML chart generation for visualization
"""
import os
from datetime import datetime
from logs import logger

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class ChartGenerator:
    """Generate interactive HTML charts with Plotly"""

    def __init__(self, profile_config, output_config, profile):
        """
        Initialize chart generator

        Args:
            profile_config: Profile configuration dictionary
            output_config: Output configuration dictionary
            profile: Profile name
        """
        self.profile_config = profile_config
        self.output_config = output_config
        self.profile = profile

    def generate(self, signals_df):
        """
        Generate interactive HTML chart

        Args:
            signals_df: DataFrame with signals and indicators

        Returns:
            str: Path to generated HTML file or None
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Skipping HTML chart generation.")
            return None

        if not self.output_config.get("html_chart", True):
            return None

        logger.info("Generating HTML chart...")

        # Create subplots
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price & SMAs", "Volume", "RSI", "Signals"),
            row_heights=[0.5, 0.15, 0.15, 0.2],
        )

        df = signals_df

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # SMAs
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sma_fast"],
                name=f"SMA {self.profile_config['indicators']['sma_fast']}",
                line=dict(color="orange", width=1),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["sma_slow"],
                name=f"SMA {self.profile_config['indicators']['sma_slow']}",
                line=dict(color="blue", width=1),
            ),
            row=1,
            col=1,
        )

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_upper"],
                name="BB Upper",
                line=dict(color="gray", width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["bb_lower"],
                name="BB Lower",
                line=dict(color="gray", width=1, dash="dash"),
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.1)",
            ),
            row=1,
            col=1,
        )

        # Buy/Sell signals
        buys = df[df["signal"] == "BUY"]
        sells = df[df["signal"] == "SELL"]

        fig.add_trace(
            go.Scatter(
                x=buys["timestamp"],
                y=buys["close"],
                mode="markers",
                name="BUY",
                marker=dict(color="green", size=15, symbol="triangle-up"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sells["timestamp"],
                y=sells["close"],
                mode="markers",
                name="SELL",
                marker=dict(color="red", size=15, symbol="triangle-down"),
            ),
            row=1,
            col=1,
        )

        # Volume
        colors = [
            "red" if row["close"] < row["open"] else "green" for _, row in df.iterrows()
        ]
        fig.add_trace(
            go.Bar(
                x=df["timestamp"],
                y=df["volume"],
                name="Volume",
                marker=dict(color=colors),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["volume_ma"],
                name="Volume MA",
                line=dict(color="orange", width=1),
            ),
            row=2,
            col=1,
        )

        # RSI
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["rsi"],
                name="RSI",
                line=dict(color="purple", width=1),
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        # Signal timeline
        signal_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        df["signal_numeric"] = df["signal"].map(signal_map)
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["signal_numeric"],
                mode="lines",
                name="Signal",
                line=dict(color="blue", width=2),
            ),
            row=4,
            col=1,
        )

        # Update layout
        fig.update_layout(
            title=f"{self.profile} Strategy - {self.profile_config['market']['symbol']} {self.profile_config['market']['timeframe']}",
            xaxis_rangeslider_visible=False,
            height=1200,
            showlegend=True,
        )

        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Signal", row=4, col=1)

        # Save
        output_dir = self.output_config["base_path"]
        html_path = os.path.join(output_dir, "chart.html")
        fig.write_html(html_path)
        logger.info(f"HTML chart saved to {html_path}")

        return html_path
