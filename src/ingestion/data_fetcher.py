"""
Data fetching and loading utilities
"""
import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta
from logs import logger


class DataFetcher:
    """Fetch historical OHLCV data from exchanges"""

    @staticmethod
    def fetch_ohlcv(symbol, timeframe, days):
        """
        Fetch historical OHLCV data from Binance

        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            timeframe: Timeframe string (e.g., "1h", "15m")
            days: Number of days of historical data to fetch

        Returns:
            pd.DataFrame: DataFrame with OHLCV data indexed by timestamp
        """
        logger.info(f"Fetching {days} days of {timeframe} data for {symbol}")

        # Parse timeframe
        if "h" in timeframe:
            resample_minutes = int(timeframe.replace("h", "")) * 60
        elif "m" in timeframe:
            resample_minutes = int(timeframe.replace("m", ""))
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")

        # Initialize exchange
        exchange = ccxt.binance({"enableRateLimit": True})
        since = exchange.parse8601((datetime.now() - timedelta(days=days)).isoformat())
        all_ohlcv = []

        # Fetch data in batches
        while True:
            ohlcv = exchange.fetch_ohlcv(symbol, "1m", since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 60000
            if len(ohlcv) < 1000:
                break
            time.sleep(exchange.rateLimit / 1000)

        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        # Resample if needed
        if resample_minutes > 1:
            df = (
                df.resample(f"{resample_minutes}min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )

        logger.info(f"Fetched {len(df)} candles")
        return df
