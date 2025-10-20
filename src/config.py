from types import SimpleNamespace

COIN = "bitcoin"
SYMBOL = "BTC"
MASTER = "C:/Users/rastr/Documents/trading-bot-project/data/master.csv"
API_URL = "https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
FILES_DIR = "C:/Users/rastr/Documents/trading-bot-project/data/temp"
NEWS_NEGATIVE_RATIO = 0.6
SENTIMENT_RULES = {
    "very_negative_compound": -0.6,
    "moderate_negative_compound": -0.25,
    "moderate_positive_compound": 0.25,
    "very_positive_compound": 0.6,
    "very_negative_ratio": NEWS_NEGATIVE_RATIO,
    "moderate_negative_ratio": 0.4,
}
DAYS = 14
DATA_PATH = "data/raw/bitcoin.csv"
MA_SHORT_PERIOD = "3h"
MA_LONG_PERIOD = "48h"
GRANULARITY = "10min"
CURRENCY = "usd"
TRAIN_FPATH = "C:/Users/rastr/Documents/trading-bot-project/data/train/train.csv"
DISPLAY_NEWS = False
NEWS_HRS = 3


AGGRESSIVE = SimpleNamespace(
    BB_WINDOW=14,
    BB_STDEV=1.5,
    SMA_FAST=5,
    SMA_SLOW=20,
    EMA_SHORT=8,
    EMA_LONG=30,
    RSI=7,
    RSI_LOW=35,
    RSI_HIGH=50,
    MIN_BUY_VOTES=2,
    MIN_SELL_VOTES=2
)
MODERATE = SimpleNamespace(
    BB_WINDOW=20,
    BB_STDEV=1.75,
    SMA_FAST=8,
    SMA_SLOW=25,
    EMA_SHORT=12,
    EMA_LONG=60,
    RSI=14,
    RSI_LOW=30,
    RSI_HIGH=60,
    MIN_BUY_VOTES=2,
    MIN_SELL_VOTES=2
)
CONSERVATIVE = SimpleNamespace(
    BB_WINDOW=25,
    BB_STDEV=2.5,
    SMA_FAST=10,
    SMA_SLOW=40,
    EMA_SHORT=15,
    EMA_LONG=75,
    RSI=25,
    RSI_LOW=20,
    RSI_HIGH=80,
    MIN_BUY_VOTES=3,
    MIN_SELL_VOTES=3
)

PROFILES = {
    "AGGRESSIVE": AGGRESSIVE,
    "MODERATE": MODERATE,
    "CONSERVATIVE": CONSERVATIVE,
}

ACTIVE_PROFILE = "MODERATE"
P = PROFILES[ACTIVE_PROFILE]

RSI = P.RSI
EMA_SHORT = P.EMA_SHORT
EMA_LONG = P.EMA_LONG
SMA_SLOW = P.SMA_SLOW
SMA_FAST = P.SMA_FAST
BB_WINDOW = P.BB_WINDOW
BB_STDEV = P.BB_STDEV
RSI_LOW = P.RSI_LOW
RSI_HIGH = P.RSI_HIGH
MIN_SELL_VOTES = P.MIN_SELL_VOTES
MIN_BUY_VOTES = P.MIN_BUY_VOTES