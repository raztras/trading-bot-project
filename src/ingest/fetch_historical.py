import os
import time
import ccxt
import pandas as pd
from logs import logger
from dotenv import load_dotenv
import src.config as cfg

load_dotenv()


def fetch_historical_prices(days, resample_minutes=5, exchange_id="binance"):
    """
    Fetch OHLCV from an exchange via ccxt. Defaults to Binance BTC/USDT.
    Returns a DataFrame indexed by naive Europe/London timestamps with columns
    open, high, low, close.
    Install: pip install ccxt
    """
    symbol = f"{cfg.SYMBOL}/USDT"

    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"enableRateLimit": True})

    timeframe = f"{int(resample_minutes)}m"
    now_ms = int(time.time() * 1000)
    since_ms = now_ms - int(days * 86400 * 1000)

    all_ohlcv = []
    limit = 1000

    while True:
        ohlcv = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        since_ms = last_ts + int(resample_minutes * 60 * 1000)
        if last_ts >= now_ms - 1:
            break
        if len(ohlcv) < limit:
            break

    if not all_ohlcv:
        logger.warning("No OHLCV returned from %s for %s", exchange_id, symbol)
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    df = pd.DataFrame(
        all_ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["timestamp"] = (
        df["timestamp"].dt.tz_convert("Europe/London").dt.tz_localize(None)
    )
    df = df.set_index("timestamp").sort_index()

    logger.info(
        "Fetched %d records from %s %s (%s)",
        len(df),
        exchange_id,
        symbol,
        timeframe,
    )
    return df[["open", "high", "low", "close"]]


def save_to_files(df, coin, currency, days):
    """
    Orchestrates resampling and metric calculation, then writes CSV for the
    requested granularity.
    Returns the enriched DataFrame (df_out) for further use.
    """
    os.makedirs(cfg.FILES_DIR, exist_ok=True)
    filename = f"{coin}_{currency}_{days}.csv"
    filepath = os.path.join(cfg.FILES_DIR, filename)

    df.to_csv(filepath)
    logger.info(f"Saved {coin} candlestick, MA, std dev, and signal data to {filepath}")
    return df


def save_to_master(coin=cfg.COIN, currency="usd", days=10):
    """
    Append the last row of the saved per-coin CSV to the master file.
    """
    filename = f"{coin}_{currency}_{days}.csv"
    filepath = os.path.join(cfg.FILES_DIR, filename)
    if not os.path.exists(filepath):
        logger.warning("Per-coin file not found, skipping master append: %s", filepath)
        return

    os.makedirs(os.path.dirname(cfg.MASTER), exist_ok=True)

    df = pd.read_csv(filepath)
    last_df = df.tail(1).copy().reset_index(drop=True)

    write_header = not os.path.exists(cfg.MASTER)
    last_df.to_csv(cfg.MASTER, mode="a", index=False, header=write_header)
    return logger.info(f"Appended latest row to master file: {cfg.MASTER}")
