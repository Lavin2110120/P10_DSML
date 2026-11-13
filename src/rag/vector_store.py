"""Gestionnaire d'index FAISS adapté aux dimensions des embeddings Gemini (768)."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src import ssl_bypass
from src.config import FAISS_INDEX_FILE, METADATA_FILE, EMBEDDING_DIM
from src.llm.gemini_client import GeminiClient

import time
from google.genai import errors
import faiss
import json
import numpy as np
from typing import List, Dict, Any

class VectorStoreManager:
    def __init__(self):
        self.client = GeminiClient()
        self.dimension = EMBEDDING_DIM  # 768
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def create_index_from_documents(self, documents: List[Dict[str, Any]]):
        """Prend une liste de dicts [{'content': '...', 'metadata': {...}}] et crée l'index."""
        texts = [
            doc.get("content") or doc.get("text") or doc.get("page_content") or ""
            for doc in documents
        ]
        print(f"Génération des embeddings pour {len(texts)} segments...")

        batch_size = 20
        pause_s = 20
        cache_file = Path(str(FAISS_INDEX_FILE) + ".partial.npy")

        # Reprise : recharge les embeddings déjà calculés
        all_embeddings: List[List[float]] = []
        if cache_file.exists():
            all_embeddings = np.load(cache_file).tolist()
            print(f"Reprise : {len(all_embeddings)} embeddings déjà calculés.")

        for i in range(len(all_embeddings), len(texts), batch_size):
            batch = texts[i:i + batch_size]
            while True:
                try:
                    embs = self.client.get_embeddings(batch, task_type="retrieval_document")
                    break
                except errors.ClientError as e:
                    if e.code == 429:
                        print("Quota atteint, attente de 60 s...")
                        time.sleep(60)
                    else:
                        raise
            all_embeddings.extend(embs)
            np.save(cache_file, np.array(all_embeddings, dtype="float32"))
            print(f"  {len(all_embeddings)}/{len(texts)}")
            time.sleep(pause_s)

        embs_np = np.array(all_embeddings, dtype="float32")
        faiss.normalize_L2(embs_np)

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embs_np)
        self.metadata = documents

        self.save()
        cache_file.unlink(missing_ok=True)
        print(f"Index créé avec succès ({self.index.ntotal} vecteurs).")

    def save(self):
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self):
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Index ou métadonnées absents. Lancez indexer.py d'abord.")
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        if self.index.d != self.dimension:
            raise ValueError(f"Dimension incompatible : {self.index.d} vs {self.dimension}. Réindexez !")
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Recherche sémantique pour une question donnée."""
        if self.index is None:
            self.load()
        q_emb = self.client.get_embeddings([query], task_type="retrieval_query")[0]
        q_np = np.array([q_emb], dtype="float32")
        faiss.normalize_L2(q_np)

        distances, indices = self.index.search(q_np, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results