# ...existing code...
import os
from pprint import pprint
import time
import feedparser
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from logs import logger

load_dotenv()

FILES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files")
SENTIMENT_RULES = {
    "very_negative_compound": -0.6,
    "moderate_negative_compound": -0.25,
    "moderate_positive_compound": 0.25,
    "very_positive_compound": 0.6,
    "very_negative_ratio": 0.6,
    "moderate_negative_ratio": 0.4,
}

def fetch_news_sentiment(
    query="Bitcoin", hours=1, max_articles=5000, display=False
):
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://cryptoslate.com/feed/",
        "https://news.bitcoin.com/feed/",
        "https://bitcoinmagazine.com/feed",
        "https://cryptonews.com/news/feed/",
        "https://www.theblock.co/latest/feed",
        "https://blog.coinbase.com/feed",
        "https://blog.kraken.com/feed",
    ]

    now_utc = datetime.now(timezone.utc)
    since_utc = now_utc - timedelta(hours=hours)

    collected = []
    seen = set()
    q_lower = (query or "").lower()
    extra_terms = ["bitcoin", "btc", "crypto", "cryptocurrency", "BTC"]

    for url in feeds:
        try:
            d = feedparser.parse(url)
        except Exception as e:
            logger.debug("feed parse failed %s: %s", url, e)
            continue

        for entry in getattr(d, "entries", []):
            pub = None
            if getattr(entry, "published_parsed", None):
                pub = datetime.fromtimestamp(
                    time.mktime(entry.published_parsed), tz=timezone.utc
                )
            elif getattr(entry, "updated_parsed", None):
                pub = datetime.fromtimestamp(
                    time.mktime(entry.updated_parsed), tz=timezone.utc
                )
            else:
                # skip if no time
                continue

            if pub < since_utc:
                continue

            title = entry.get("title", "") or ""
            summary = (
                entry.get("summary", "") or entry.get("description", "") or ""
            )
            text = f"{title} {summary}".strip()
            link = entry.get("link")

            if not link or link in seen:
                continue

            text_l = text.lower()
            if q_lower and q_lower not in text_l:
                if not any(t in text_l for t in extra_terms):
                    continue

            seen.add(link)
            collected.append(
                {
                    "title": title,
                    "summary": summary,
                    "text": text,
                    "published": pub,
                    "link": link,
                }
            )
            if len(collected) >= max_articles:
                break
        if len(collected) >= max_articles:
            break

    logger.info("Collected %d RSS items", len(collected))
    if not collected:
        return {
            "total": 0,
            "negative": 0,
            "neutral": 0,
            "positive": 0,
            "negative_ratio": 0.0,
            "avg_compound": 0.0,
        }

    analyzer = SentimentIntensityAnalyzer()
    neg = neu = pos = 0
    comp_sum = 0.0

    if display:
        logger.info(pprint(collected))
    for item in collected:
        vs = analyzer.polarity_scores(item["text"])
        comp = vs["compound"]
        comp_sum += comp
        if comp <= -0.05:
            neg += 1
        elif comp >= 0.05:
            pos += 1
        else:
            neu += 1

    total = len(collected)
    denom = total
    if denom <= 0:
        negative_ratio = 0.0
        avg_compound = 0.0
    else:
        negative_ratio = neg / denom
        avg_compound = comp_sum / denom

    return {
        "total": total,
        "negative": neg,
        "neutral": neu,
        "positive": pos,
        "negative_ratio": negative_ratio,
        "avg_compound": avg_compound,
    }


def get_sentiment_label(
    avg_compound,
    negative_ratio,
    non_neutral,
    min_non_neutral=1,
    thresholds=None,
):
    """
    Return one of: 'negative', 'moderately_negative', 'neutral',
    'moderately_positive', 'positive'.

    Rules:
      - require at least `min_non_neutral` non-neutral items to act,
        otherwise return 'neutral'.
      - check strong signals first, then moderate ones.
    """
    if thresholds is None:
        thresholds = SENTIMENT_RULES
    if non_neutral < max(1, min_non_neutral):
        return "neutral"

    if avg_compound >= thresholds["very_positive_compound"]:
        return "positive"
    
    if avg_compound >= thresholds["moderate_positive_compound"]:
        return "moderately_positive"

    if (
        avg_compound <= thresholds["very_negative_compound"]
        or negative_ratio >= thresholds["very_negative_ratio"]
    ):
        return "negative"
    if (
        avg_compound <= thresholds["moderate_negative_compound"]
        or negative_ratio >= thresholds["moderate_negative_ratio"]
    ):
        return "moderately_negative"

    return "neutral"

def get_sentiment(news_hours, display):
    stats = fetch_news_sentiment(hours=news_hours, display=display)
    logger.info(
        f"news negative_ratio={stats['negative_ratio']:.2f}, articles={stats['total']}"
    )

    avg_compound = float(stats.get("avg_compound", 0.0))
    negative_ratio = float(stats.get("negative_ratio", 0.0))
    total = int(stats.get("total", 0))
    neutral = int(stats.get("neutral", 0))
    non_neutral = max(0, total - neutral)

    sentiment_label = get_sentiment_label(
        avg_compound,
        negative_ratio,
        non_neutral,
        min_non_neutral=1,
        thresholds=SENTIMENT_RULES,
    )

    return stats, sentiment_label

if __name__ == "__main__":
    stats = fetch_news_sentiment(hours=2)
    print(stats)

