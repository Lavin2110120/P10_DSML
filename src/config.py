from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

# This module lives in <project>/src/config.py. Resolve .env from the project,
# not from the process working directory (python -m may be launched elsewhere).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = PROJECT_ROOT / ".env"
if load_dotenv is not None and _ENV_FILE.is_file():
    load_dotenv(dotenv_path=_ENV_FILE, override=False)

# Apply the process-wide patch before importing SDK/client modules elsewhere.
try:
    from .ssl_bypass import disable_ssl_verification
except ImportError:
    from src.ssl_bypass import disable_ssl_verification

DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "0").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
_SSL_BYPASS_ACTIVE = disable_ssl_verification()

APP_TITLE = os.getenv("APP_TITLE", "NBA Analyst AI")
NAME = os.getenv("APP_NAME", "NBA")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "test-key")
MODEL_NAME = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
EMBEDDING_MODEL = os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
SEARCH_K = int(os.getenv("SEARCH_K", "5"))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
LOGS_DIR = PROJECT_ROOT / "logs"
FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index.idx"
DOCUMENT_CHUNKS_FILE = VECTOR_DB_DIR / "document_chunks.pkl"
LOG_FILE = LOGS_DIR / "app.log"
INPUT_DIR = RAW_DATA_DIR
PROCESSED_DIR = PROCESSED_DATA_DIR

for _path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_DB_DIR, LOGS_DIR, DATABASE_DIR):
    _path.mkdir(parents=True, exist_ok=True)


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        try:
            file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            pass
    return logger
