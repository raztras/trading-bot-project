import os

import pandas as pd
from src.archive.metrics import add_scalping_metrics
from src.archive.charts import plot_chart
from src.archive.fetch_historical import (
    fetch_historical_prices,
    save_to_files,
    save_to_master,
)
from src.archive.rss_scrape import get_sentiment
from logs import logger
import src.config as cfg
from src.simulator.sim_trades import simulate_trades


def main(
    news_hours=cfg.NEWS_HRS,
    days=cfg.DAYS,
    currency=cfg.CURRENCY,
    display=cfg.DISPLAY_NEWS,
    granularity=cfg.GRANULARITY,
    # Optional advanced features
    trailing_stop_pct=None,
    partial_profit_pct=None,
    partial_profit_size=0.5,
):
    if os.path.exists(cfg.FILES_DIR):
        for filename in os.listdir(cfg.FILES_DIR):
            file_path = os.path.join(cfg.FILES_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(cfg.FILES_DIR, exist_ok=True)

    df = fetch_historical_prices(days=days)
    df = add_scalping_metrics(df)

    # Get sentiment data
    try:
        stats, sentiment_label = get_sentiment(news_hours, display)
        df["news_articles"] = int(stats.get("total", 0))
        df["sentiment"] = sentiment_label
        df["news_negative_ratio"] = round(float(stats.get("negative_ratio", 0.0)), 4)
        df["news_avg_compound"] = round(float(stats.get("avg_compound", 0.0)), 4)
        logger.info(f"Sentiment analysis complete: {sentiment_label} (articles: {stats.get('total', 0)})")
    except Exception as e:
        logger.warning(f"Sentiment analysis failed: {e}. Continuing without sentiment data.")
        df["news_articles"] = 0
        df["sentiment"] = "neutral"
        df["news_negative_ratio"] = 0.0
        df["news_avg_compound"] = 0.0

    # Run simulation with risk management settings from active profile
    logger.info(f"Running simulation with {cfg.ACTIVE_PROFILE} profile settings:")
    logger.info(f"  - Position Sizing: {cfg.P.POSITION_SIZING_METHOD}")
    logger.info(f"  - Max Position Size: {cfg.P.MAX_POSITION_SIZE * 100:.0f}%")
    logger.info(f"  - Max Daily Loss: {cfg.P.MAX_DAILY_LOSS * 100:.1f}%")
    logger.info(f"  - Max Consecutive Losses: {cfg.P.MAX_CONSECUTIVE_LOSSES}")

    # Log advanced features if enabled
    if trailing_stop_pct is not None:
        logger.info(f"  - Trailing Stop: {trailing_stop_pct * 100:.2f}%")
    if partial_profit_pct is not None:
        logger.info(f"  - Partial Profit: {partial_profit_pct * 100:.2f}% (selling {partial_profit_size * 100:.0f}%)")

    df, trades_df, summary = simulate_trades(
        df,
        start_cash=100_000.0,
        buy_fee=0.001,
        sell_fee=0.001,
        target_pct=0.01,  # 1% take profit
        stop_loss_pct=0.0075,  # 0.75% stop loss
        # Risk management from profile
        max_position_size=cfg.P.MAX_POSITION_SIZE,
        max_daily_loss=cfg.P.MAX_DAILY_LOSS,
        max_consecutive_losses=cfg.P.MAX_CONSECUTIVE_LOSSES,
        position_sizing_method=cfg.P.POSITION_SIZING_METHOD,
        # Advanced features (passed from CLI)
        trailing_stop_pct=trailing_stop_pct,
        partial_profit_pct=partial_profit_pct,
        partial_profit_size=partial_profit_size,
    )

    # Log summary
    logger.info(f"Simulation complete:")
    logger.info(f"  - Total Trades: {summary['n_trades']}")
    logger.info(f"  - Win Rate: {summary['win_rate']:.2%}")
    logger.info(f"  - Total Return: {summary['total_net_ret']:.2%}")
    logger.info(f"  - Final Equity: ${summary['end_equity']:,.2f}")

    df.to_csv(cfg.TRAIN_FPATH)
    logger.info(f"Saved results to {cfg.TRAIN_FPATH}")

    save_to_files(df, cfg.COIN, currency, days)
    plot_chart(df)
    # save_to_master(coin=cfg.COIN, currency=currency, days=days)
    logger.info(f"Completed {cfg.COIN} ingestion")
