"""Script d'indexation vectorielle dans FAISS."""

import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src import ssl_bypass
from src.config import DATA_PROCESSED_DIR
from src.rag.vector_store import VectorStoreManager

def run_indexing():
    input_file = DATA_PROCESSED_DIR / "documents.json"
    if not input_file.exists():
        print(f"❌ Erreur : {input_file} non trouvé. Exécutez d'abord data_loader.py !")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    print(f"Chargement de {len(documents)} documents depuis {input_file}...")
    manager = VectorStoreManager()
    manager.create_index_from_documents(documents)
    print("✅ Indexation FAISS avec Gemini terminée !")

if __name__ == "__main__":
    run_indexing()