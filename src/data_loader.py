import os

import pandas as pd
from src.ingest.metrics import add_scalping_metrics
from src.ingest.charts import plot_chart
from src.ingest.fetch_historical import (
    fetch_historical_prices,
    save_to_files,
    save_to_master,
)
from src.ingest.rss_scrape import get_sentiment
from logs import logger
import src.config as cfg
from src.simulator.sim_trades import simulate_trades


def main(
    news_hours=cfg.NEWS_HRS,
    days=cfg.DAYS,
    currency=cfg.CURRENCY,
    display=cfg.DISPLAY_NEWS,
    granularity=cfg.GRANULARITY,
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
    df, trades_df, summary = simulate_trades(df)
    df.to_csv(cfg.TRAIN_FPATH)

    # stats, sentiment_label = get_sentiment(news_hours, display)

    # df["news_articles"] = int(stats.get("total", 0))
    # df["sentiment"] = sentiment_label
    # df["news_negative_ratio"] = round(float(stats.get("negative_ratio", 0.0)), 4)
    # df["news_avg_compound"] = round(float(stats.get("avg_compound", 0.0)), 4)

    save_to_files(df, cfg.COIN, currency, days)
    plot_chart(df)
    # save_to_master(coin=cfg.COIN, currency=currency, days=days)
    logger.info(f"Completed {cfg.COIN} ingestion")
