class TechnicalIndicators:
    """Calculate technical indicators for trading"""

    @staticmethod
    def calculate_all(df, indicator_config):
        """
        Calculate all technical indicators

        Args:
            df: DataFrame with OHLCV data
            indicator_config: Dictionary with indicator parameters

        Returns:
            pd.DataFrame: DataFrame with all indicators added
        """
        df = df.copy()

        # SMA - shift by 1 to prevent data leakage
        df["sma_fast"] = df["close"].rolling(indicator_config["sma_fast"]).mean().shift(1)
        df["sma_slow"] = df["close"].rolling(indicator_config["sma_slow"]).mean().shift(1)
        df["sma_cross_up"] = (df["sma_fast"] > df["sma_slow"]) & (
            df["sma_fast"].shift(1) <= df["sma_slow"].shift(1)
        )
        df["sma_cross_down"] = (df["sma_fast"] < df["sma_slow"]) & (
            df["sma_fast"].shift(1) >= df["sma_slow"].shift(1)
        )

        # Volume - shift by 1 to prevent data leakage
        df["volume_ma"] = df["volume"].rolling(indicator_config["volume_ma_period"]).mean().shift(1)
        df["high_volume"] = df["volume"].shift(1) > df["volume_ma"]
        df["volume_ratio"] = df["volume"].shift(1) / df["volume_ma"]

        # RSI - shift by 1 to prevent data leakage
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(indicator_config["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(indicator_config["rsi_period"]).mean()
        rs = gain / loss
        df["rsi"] = (100 - (100 / (1 + rs))).shift(1)

        # Bollinger Bands - shift by 1 to prevent data leakage
        df["bb_middle"] = df["close"].rolling(indicator_config["bb_period"]).mean().shift(1)
        bb_std = df["close"].rolling(indicator_config["bb_period"]).std().shift(1)
        df["bb_upper"] = df["bb_middle"] + (bb_std * indicator_config["bb_std"])
        df["bb_lower"] = df["bb_middle"] - (bb_std * indicator_config["bb_std"])
        df["bb_position"] = (df["close"].shift(1) - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Volatility - shift by 1 to prevent data leakage
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std().shift(1)

        # Momentum - shift by 1 to prevent data leakage
        df["momentum_5"] = df["close"].pct_change(5).shift(1)
        df["momentum_10"] = df["close"].pct_change(10).shift(1)

        # ML Features - use shifted close to prevent data leakage
        df["price_to_sma_fast"] = df["close"].shift(1) / df["sma_fast"] - 1
        df["price_to_sma_slow"] = df["close"].shift(1) / df["sma_slow"] - 1
        df["sma_ratio"] = df["sma_fast"] / df["sma_slow"] - 1

        return df.dropna()
