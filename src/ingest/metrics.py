import numpy as np
import pandas as pd
from src.config import (
    RSI,
    EMA_SHORT,
    EMA_LONG,
    SMA_SLOW,
    SMA_FAST,
    BB_WINDOW,
    BB_STDEV,
    RSI_HIGH,
    RSI_LOW,
    MIN_BUY_VOTES,
    MIN_SELL_VOTES,
)


def add_sma(
    df: pd.DataFrame, window=20, price_col: str = "close", col_name: str | None = None
) -> pd.DataFrame:
    """
    Add Simple Moving Average over a period count or time-based window (e.g., '15T').
    """
    dfx = df.copy()
    name = col_name or f"sma_{window}"
    if isinstance(window, str):
        dfx[name] = dfx[price_col].rolling(window=window).mean().round(2)
    else:
        dfx[name] = (
            dfx[price_col]
            .rolling(window=int(window), min_periods=int(window))
            .mean()
            .round(2)
        )
    return dfx


def add_ema(
    df: pd.DataFrame, span=20, price_col: str = "close", col_name: str | None = None
) -> pd.DataFrame:
    """
    Add Exponential Moving Average.
    """
    dfx = df.copy()
    name = col_name or f"ema_{span}"
    dfx[name] = dfx[price_col].ewm(span=int(span), adjust=False).mean().round(2)
    return dfx


def add_std(
    df: pd.DataFrame, window=20, price_col: str = "close", col_name: str | None = None
) -> pd.DataFrame:
    """
    Add rolling standard deviation over a period count or time-based window.
    """
    dfx = df.copy()
    name = col_name or f"std_{window}"
    if isinstance(window, str):
        dfx[name] = dfx[price_col].rolling(window=window).std(ddof=0).round(2)
    else:
        dfx[name] = (
            dfx[price_col]
            .rolling(window=int(window), min_periods=int(window))
            .std(ddof=0)
            .round(2)
        )
    return dfx


def add_bollinger_bands(
    df: pd.DataFrame,
    window=BB_WINDOW,
    num_std: float = BB_STDEV,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Add Bollinger Bands (middle=SMA, upper/lower = SMA ± num_std * StdDev).
    """
    dfx = df.copy()
    dfx = add_sma(dfx, window=window, price_col=price_col, col_name="bb_mid").round(2)
    dfx = add_std(dfx, window=window, price_col=price_col, col_name="bb_std").round(2)
    dfx["bb_upper"] = (dfx["bb_mid"] + num_std * dfx["bb_std"]).round(2)
    dfx["bb_lower"] = (dfx["bb_mid"] - num_std * dfx["bb_std"]).round(2)
    return dfx


def add_rsi(
    df: pd.DataFrame,
    rsi_window: int | str = 14,
    price_col: str = "close",
    col_name: str | None = None,
) -> pd.DataFrame:
    """
    Add Relative Strength Index (RSI).
    - If window is int: uses Wilder's smoothing (EMA with alpha=1/window).
    - If window is str (e.g., '1H'): uses SMA over a time-based rolling window.
    """
    dfx = df.copy()

    delta = dfx[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    if isinstance(rsi_window, str):
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
    else:
        w = int(rsi_window)
        avg_gain = gain.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()
        avg_loss = loss.ewm(alpha=1 / w, adjust=False, min_periods=w).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    both_zero = (avg_gain == 0) & (avg_loss == 0)
    no_loss = (avg_loss == 0) & (~both_zero)
    no_gain = (avg_gain == 0) & (~both_zero)

    rsi = rsi.mask(both_zero, 50.0)
    rsi = rsi.mask(no_loss, 100.0)
    rsi = rsi.mask(no_gain, 0.0)

    dfx[col_name] = rsi.round(2)
    return dfx


def add_trade_tags(
    df: pd.DataFrame,
    price_col: str = "close",
    rsi_low: int = 30,
    rsi_high: int = 70,
    ensure_indicators: bool = True,
) -> pd.DataFrame:
    """
    Tag BUY/SELL based on RSI, SMAs, EMAs, and Bollinger Bands.
    Produces: buy (bool), sell (bool), signal ('BUY'|'SELL'|''), buy_score, sell_score
    """
    need_cols = {
        "sma_fast",
        "sma_slow",
        "ema_short",
        "ema_long",
        "rsi",
        "bb_upper",
        "bb_mid",
        "bb_lower",
        price_col,
    }
    dfx = df.copy()

    # Ensure indicators exist (uses your configured windows)
    if ensure_indicators and not need_cols.issubset(dfx.columns):
        dfx = add_scalping_metrics(dfx, price_col=price_col)

    # Short aliases
    sf, ss = dfx["sma_fast"], dfx["sma_slow"]
    es, el = dfx["ema_short"], dfx["ema_long"]
    rsi = dfx["rsi"]
    cu = dfx["bb_upper"]
    cm = dfx["bb_mid"]
    cl = dfx["bb_lower"]
    close = dfx[price_col]

    s1 = dfx.shift(1)

    # Helpers: cross detection
    def cross_over(a, b):  # a crosses above b
        return (a > b) & (s1[a.name] <= s1[b.name])

    def cross_under(a, b):  # a crosses below b
        return (a < b) & (s1[a.name] >= s1[b.name])



    # BUY confirmations
    bull_cross = cross_over(sf, ss) | cross_over(es, el)
    rsi_rebound = (s1["rsi"] < rsi_low) & (rsi >= rsi_low)
    bb_reentry = (s1[price_col] < s1["bb_lower"]) & (close >= cl)
    above_mid = close > cm
    bb_break_lower = (s1[price_col] >= s1["bb_lower"]) & (close < cl)

    # SELL confirmations
    bear_cross = cross_under(sf, ss) | cross_under(es, el)
    rsi_fade = (s1["rsi"] > rsi_high) & (rsi <= rsi_high)
    bb_reject = (s1[price_col] > s1["bb_upper"]) & (close <= cu)
    below_mid = close < cm
    bb_break_upper = (s1[price_col] <= s1["bb_upper"]) & (close > cu)

    # Votes
    buy_votes = (
        bull_cross.astype(int)
        + (rsi_rebound.astype(int))
        + bb_reentry.astype(int)
        + above_mid.astype(int)
        + 2*(bb_break_lower.astype(int))
    )
    sell_votes = (
        bear_cross.astype(int)
        + (rsi_fade.astype(int))
        + bb_reject.astype(int)
        + below_mid.astype(int)
        + 2*(bb_break_upper.astype(int))
    )

    # # Trend filters
    # trend_up = (es > el) | (sf > ss)
    # trend_dn = (es < el) | (sf < ss)

    # buy = trend_up & (buy_votes >= MIN_BUY_VOTES)
    # sell = trend_dn & (sell_votes >= MIN_SELL_VOTES)

    buy = buy_votes >= MIN_BUY_VOTES
    sell = sell_votes >= MIN_SELL_VOTES

    # Prevent simultaneous signals on same bar
    sell = sell & ~buy

    dfx["buy"] = buy
    dfx["sell"] = sell
    dfx["buy_score"] = buy_votes
    dfx["sell_score"] = sell_votes
    dfx["signal"] = np.where(dfx["buy"], "BUY", np.where(dfx["sell"], "SELL", ""))

    return dfx


def add_scalping_metrics(
    df: pd.DataFrame,
    sma_fast=SMA_FAST,
    sma_slow=SMA_SLOW,
    ema_long=EMA_LONG,
    ema_short=EMA_SHORT,
    rsi_window=RSI,
    bb_window=BB_WINDOW,
    bb_std=BB_STDEV,
    price_col: str = "close",
    rsi_low=RSI_LOW,
    rsi_high=RSI_HIGH,
) -> pd.DataFrame:
    """
    Convenience helper to append common short-term indicators for scalping.
    """
    dfx = df.copy()
    dfx = add_sma(dfx, window=sma_fast, price_col=price_col, col_name="sma_fast")
    dfx = add_sma(dfx, window=sma_slow, price_col=price_col, col_name="sma_slow")
    dfx = add_ema(dfx, span=ema_short, price_col=price_col, col_name="ema_short")
    dfx = add_ema(dfx, span=ema_long, price_col=price_col, col_name="ema_long")
    dfx = add_rsi(dfx, rsi_window=rsi_window, col_name="rsi")
    dfx = add_bollinger_bands(
        dfx,
        window=bb_window,
        num_std=bb_std,
        price_col=price_col,
    )
    dfx = add_trade_tags(dfx, price_col=price_col, rsi_low=rsi_low, rsi_high=rsi_high)
    return dfx
