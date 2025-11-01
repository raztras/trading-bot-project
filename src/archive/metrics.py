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


def add_adx(
    df: pd.DataFrame,
    window: int = 14,
    price_col_high: str = "high",
    price_col_low: str = "low",
    price_col_close: str = "close",
    col_name: str = "adx",
) -> pd.DataFrame:
    """
    Add Average Directional Index (ADX) to measure trend strength.
    ADX values:
      - < 20: Weak/no trend (ranging market)
      - 20-25: Emerging trend
      - 25-50: Strong trend
      - > 50: Very strong trend
    """
    dfx = df.copy()

    high = dfx[price_col_high]
    low = dfx[price_col_low]
    close = dfx[price_col_close]

    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # Smooth with Wilder's method (EMA with alpha=1/window)
    atr = tr.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1/window, adjust=False, min_periods=window).mean() / atr

    # Calculate ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/window, adjust=False, min_periods=window).mean()

    dfx[col_name] = adx.round(2)
    dfx[f"{col_name}_plus_di"] = plus_di.round(2)
    dfx[f"{col_name}_minus_di"] = minus_di.round(2)

    return dfx


def add_stochastic(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
    price_col_high: str = "high",
    price_col_low: str = "low",
    price_col_close: str = "close",
    col_name_k: str = "stoch_k",
    col_name_d: str = "stoch_d",
) -> pd.DataFrame:
    """
    Add Stochastic Oscillator (%K and %D) for mean reversion.
    Values range 0-100:
      - > 80: Overbought
      - < 20: Oversold
    """
    dfx = df.copy()

    close = dfx[price_col_close]
    low_min = dfx[price_col_low].rolling(window=k_window, min_periods=k_window).min()
    high_max = dfx[price_col_high].rolling(window=k_window, min_periods=k_window).max()

    # %K (fast stochastic)
    stoch_k = 100 * (close - low_min) / (high_max - low_min)
    stoch_k = stoch_k.fillna(50)  # Neutral when range is zero

    # %D (slow stochastic - SMA of %K)
    stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

    dfx[col_name_k] = stoch_k.round(2)
    dfx[col_name_d] = stoch_d.round(2)

    return dfx


def add_williams_r(
    df: pd.DataFrame,
    window: int = 14,
    price_col_high: str = "high",
    price_col_low: str = "low",
    price_col_close: str = "close",
    col_name: str = "williams_r",
) -> pd.DataFrame:
    """
    Add Williams %R for mean reversion.
    Values range -100 to 0:
      - > -20: Overbought
      - < -80: Oversold
    """
    dfx = df.copy()

    close = dfx[price_col_close]
    high_max = dfx[price_col_high].rolling(window=window, min_periods=window).max()
    low_min = dfx[price_col_low].rolling(window=window, min_periods=window).min()

    williams_r = -100 * (high_max - close) / (high_max - low_min)
    williams_r = williams_r.fillna(-50)  # Neutral when range is zero

    dfx[col_name] = williams_r.round(2)

    return dfx


def detect_market_regime(
    df: pd.DataFrame,
    adx_col: str = "adx",
    adx_trending_threshold: float = 25.0,
    vol_window: int = 20,
    price_col: str = "close",
    regime_col: str = "market_regime",
) -> pd.DataFrame:
    """
    Detect market regime: TRENDING or RANGING.

    Uses ADX to determine trend strength:
      - ADX >= threshold: TRENDING
      - ADX < threshold: RANGING

    Also adds volatility measure for additional context.
    """
    dfx = df.copy()

    # Ensure ADX exists
    if adx_col not in dfx.columns:
        dfx = add_adx(dfx)

    # Calculate volatility (rolling std of returns)
    returns = dfx[price_col].pct_change()
    volatility = returns.rolling(window=vol_window, min_periods=vol_window).std()

    # Classify regime based on ADX
    regime = pd.Series("RANGING", index=dfx.index)
    regime[dfx[adx_col] >= adx_trending_threshold] = "TRENDING"

    dfx[regime_col] = regime
    dfx["volatility"] = volatility.round(6)

    return dfx


def add_trade_tags(
    df: pd.DataFrame,
    price_col: str = "close",
    rsi_low: int = 30,
    rsi_high: int = 70,
    min_buy_votes: int = None,
    min_sell_votes: int = None,
    ensure_indicators: bool = True,
    use_regime_filter: bool = False,
    adx_threshold: float = 25.0,
    use_mean_reversion: bool = False,
    stoch_oversold: float = 20.0,
    stoch_overbought: float = 80.0,
) -> pd.DataFrame:
    """
    Tag BUY/SELL based on RSI, SMAs, EMAs, and Bollinger Bands.
    Produces: buy (bool), sell (bool), signal ('BUY'|'SELL'|''), buy_score, sell_score

    Args:
        df: DataFrame with OHLCV data
        price_col: Column name for close price
        rsi_low: RSI threshold for oversold
        rsi_high: RSI threshold for overbought
        min_buy_votes: Minimum votes to trigger BUY (defaults to global config)
        min_sell_votes: Minimum votes to trigger SELL (defaults to global config)
        ensure_indicators: Calculate indicators if missing
        use_regime_filter: If True, only trade in trending markets (ADX-based)
        adx_threshold: Minimum ADX value to consider market trending (default 25)
        use_mean_reversion: If True, add mean-reversion signals for ranging markets
        stoch_oversold: Stochastic %K oversold threshold (default 20)
        stoch_overbought: Stochastic %K overbought threshold (default 80)
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

    if ensure_indicators and not need_cols.issubset(dfx.columns):
        dfx = add_scalping_metrics(dfx, price_col=price_col)

    sf, ss = dfx["sma_fast"], dfx["sma_slow"]
    es, el = dfx["ema_short"], dfx["ema_long"]
    rsi = dfx["rsi"]
    cu = dfx["bb_upper"]
    cm = dfx["bb_mid"]
    cl = dfx["bb_lower"]
    close = dfx[price_col]

    s1 = dfx.shift(1)

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

    # Use provided vote thresholds or fall back to global config
    min_buy = min_buy_votes if min_buy_votes is not None else MIN_BUY_VOTES
    min_sell = min_sell_votes if min_sell_votes is not None else MIN_SELL_VOTES

    # Add mean-reversion signals if requested
    if use_mean_reversion:
        # Ensure stochastic indicators exist
        if "stoch_k" not in dfx.columns:
            dfx = add_stochastic(dfx)

        stoch_k = dfx["stoch_k"]

        # Mean-reversion BUY: Stochastic oversold + price below BB lower
        mean_rev_buy = (stoch_k < stoch_oversold) & (close < cl)

        # Mean-reversion SELL: Stochastic overbought + price above BB upper
        mean_rev_sell = (stoch_k > stoch_overbought) & (close > cu)

        # Add mean-reversion votes to totals
        buy_votes = buy_votes + mean_rev_buy.astype(int)
        sell_votes = sell_votes + mean_rev_sell.astype(int)

    # Generate initial signals based on votes
    buy = buy_votes >= min_buy
    sell = sell_votes >= min_sell

    # Apply regime filter if requested
    if use_regime_filter:
        # Ensure ADX and regime columns exist
        if "adx" not in dfx.columns:
            dfx = add_adx(dfx)
        if "market_regime" not in dfx.columns:
            dfx = detect_market_regime(dfx, adx_trending_threshold=adx_threshold)

        # Only allow trend-following signals in TRENDING markets
        is_trending = dfx["market_regime"] == "TRENDING"

        # Filter trend-following signals (only in trending markets)
        # Mean-reversion signals work in RANGING markets, so keep those
        if not use_mean_reversion:
            # Pure trend-following strategy: only trade when trending
            buy = buy & is_trending
            sell = sell & is_trending
        else:
            # Hybrid strategy: trend signals in trending markets, mean-reversion in ranging
            is_ranging = ~is_trending

            # Separate trend and mean-reversion components
            trend_buy = (bull_cross | (bb_break_lower)) & is_trending
            trend_sell = (bear_cross | (bb_break_upper)) & is_trending

            mean_rev_buy_final = (stoch_k < stoch_oversold) & (close < cl) & is_ranging
            mean_rev_sell_final = (stoch_k > stoch_overbought) & (close > cu) & is_ranging

            # Combine: trend signals in trending, mean-reversion in ranging
            buy = (buy & is_trending) | mean_rev_buy_final
            sell = (sell & is_trending) | mean_rev_sell_final

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
    min_buy_votes=None,
    min_sell_votes=None,
    use_regime_filter=False,
    adx_threshold=25.0,
    use_mean_reversion=False,
    stoch_oversold=20.0,
    stoch_overbought=80.0,
) -> pd.DataFrame:
    """
    Convenience helper to append common short-term indicators for scalping.

    Args:
        df: DataFrame with OHLCV data
        sma_fast: Fast SMA period
        sma_slow: Slow SMA period
        ema_short: Short EMA period
        ema_long: Long EMA period
        rsi_window: RSI period
        bb_window: Bollinger Bands window
        bb_std: Bollinger Bands standard deviation multiplier
        price_col: Column name for close price
        rsi_low: RSI oversold threshold
        rsi_high: RSI overbought threshold
        min_buy_votes: Minimum votes for BUY signal (defaults to global config)
        min_sell_votes: Minimum votes for SELL signal (defaults to global config)
        use_regime_filter: If True, only trade in trending markets
        adx_threshold: Minimum ADX for trending (default 25)
        use_mean_reversion: Add mean-reversion signals for ranging markets
        stoch_oversold: Stochastic oversold threshold (default 20)
        stoch_overbought: Stochastic overbought threshold (default 80)
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
    dfx = add_trade_tags(
        dfx,
        price_col=price_col,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        min_buy_votes=min_buy_votes,
        min_sell_votes=min_sell_votes,
        use_regime_filter=use_regime_filter,
        adx_threshold=adx_threshold,
        use_mean_reversion=use_mean_reversion,
        stoch_oversold=stoch_oversold,
        stoch_overbought=stoch_overbought,
    )
    return dfx
