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

        # SMA
        df["sma_fast"] = df["close"].rolling(indicator_config["sma_fast"]).mean()
        df["sma_slow"] = df["close"].rolling(indicator_config["sma_slow"]).mean()
        df["sma_cross_up"] = (df["sma_fast"] > df["sma_slow"]) & (
            df["sma_fast"].shift(1) <= df["sma_slow"].shift(1)
        )
        df["sma_cross_down"] = (df["sma_fast"] < df["sma_slow"]) & (
            df["sma_fast"].shift(1) >= df["sma_slow"].shift(1)
        )

        # Volume
        df["volume_ma"] = df["volume"].rolling(indicator_config["volume_ma_period"]).mean()
        df["high_volume"] = df["volume"] > df["volume_ma"]
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(indicator_config["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(indicator_config["rsi_period"]).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["bb_middle"] = df["close"].rolling(indicator_config["bb_period"]).mean()
        bb_std = df["close"].rolling(indicator_config["bb_period"]).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * indicator_config["bb_std"])
        df["bb_lower"] = df["bb_middle"] - (bb_std * indicator_config["bb_std"])
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        # Volatility
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(20).std()

        # Momentum
        df["momentum_5"] = df["close"].pct_change(5)
        df["momentum_10"] = df["close"].pct_change(10)

        # ML Features
        df["price_to_sma_fast"] = df["close"] / df["sma_fast"] - 1
        df["price_to_sma_slow"] = df["close"] / df["sma_slow"] - 1
        df["sma_ratio"] = df["sma_fast"] / df["sma_slow"] - 1

        return df.dropna()
