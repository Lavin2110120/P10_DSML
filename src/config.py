"""Configuration centralisée pour le pipeline RAG Gemini."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Racine du projet : dossier parent de src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Chargement du .env à la racine
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ==============================================================================
# Clés et Modèles Gemini
# ==============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    print("⚠️ AVERTISSEMENT : GOOGLE_API_KEY n'est pas définie dans le fichier .env !")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-001")

# Alias rétrocompatibles
MODEL_NAME = GEMINI_MODEL
EMBEDDING_MODEL_NAME = GEMINI_EMBEDDING_MODEL
EMBEDDING_DIM = 3072

# ==============================================================================
# Sécurité & SSL
# ==============================================================================
DISABLE_SSL_VERIFY = os.getenv("DISABLE_SSL_VERIFY", "0").lower() in ("1", "true", "yes")

# ==============================================================================
# Chemins du projet
# ==============================================================================
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
FAISS_INDEX_FILE = VECTOR_DB_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_DB_DIR / "metadata.json"

LOGS_DIR = PROJECT_ROOT / "logs"
DATABASE_DIR = PROJECT_ROOT / "database"

for directory in [DATA_PROCESSED_DIR, VECTOR_DB_DIR, LOGS_DIR, DATABASE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Hyperparamètres RAG
# ==============================================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SEARCH_K = 3