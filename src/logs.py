import logging
import os
from datetime import datetime

# Get the repository root (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
today_str = datetime.now().strftime("%Y-%m-%d")
LOGS_DIR = os.path.join(BASE_DIR, "log_txt")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.abspath(os.path.join(LOGS_DIR, f"{today_str}.txt"))

logger = logging.getLogger("crypto_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()

file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
file_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
))

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
))

logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.propagate = False
