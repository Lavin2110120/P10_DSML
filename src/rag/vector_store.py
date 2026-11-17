"""Gestionnaire d'index FAISS aux dimensions configurées pour les embeddings."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google.genai import errors

from src.config import EMBEDDING_DIM, FAISS_INDEX_FILE, METADATA_FILE
from src.llm.gemini_client import GeminiClient
from src.observability import span


class VectorStoreManager:
    def __init__(self) -> None:
        self.client = GeminiClient()
        self.dimension = int(EMBEDDING_DIM)
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    def _validate_embeddings(self, embeddings: list[list[float]], *, expected_count: int | None = None) -> None:
        if expected_count is not None and len(embeddings) != expected_count:
            raise ValueError(
                f"Nombre d'embeddings incohérent : reçu {len(embeddings)}, attendu {expected_count}."
            )
        for index, embedding in enumerate(embeddings):
            actual = len(embedding)
            if actual != self.dimension:
                raise ValueError(
                    f"Dimension d'embedding incohérente à l'élément {index} : "
                    f"reçu {actual}, attendu EMBEDDING_DIM={self.dimension}. "
                    "Vérifiez le modèle d'embedding/configuration et réindexez."
                )

    def create_index_from_documents(self, documents: list[dict[str, Any]]) -> None:
        """Crée l'index à partir de documents normalisés et validés."""
        texts = [
            doc.get("content") or doc.get("text") or doc.get("page_content") or ""
            for doc in documents
        ]
        if not texts:
            raise ValueError("Aucun chunk à indexer.")
        print(f"Génération des embeddings pour {len(texts)} segments...")

        batch_size = 20
        pause_s = 20
        cache_file = Path(str(FAISS_INDEX_FILE) + ".partial.npy")
        all_embeddings: list[list[float]] = []
        if cache_file.exists():
            cached = np.load(cache_file)
            if cached.ndim != 2 or cached.shape[1] != self.dimension:
                raise ValueError(
                    f"Dimension du cache d'embeddings incohérente : "
                    f"{cached.shape}, attendu (*, {self.dimension}). Supprimez le cache et relancez."
                )
            if cached.shape[0] > len(texts):
                raise ValueError("Le cache contient plus d'embeddings que de documents à indexer.")
            all_embeddings = cached.tolist()
            print(f"Reprise : {len(all_embeddings)} embeddings déjà calculés.")

        for i in range(len(all_embeddings), len(texts), batch_size):
            batch = texts[i : i + batch_size]
            while True:
                try:
                    embs = self.client.get_embeddings(batch, task_type="retrieval_document")
                    break
                except errors.ClientError as exc:
                    if exc.code == 429:
                        print("Quota atteint, attente de 60 s...")
                        time.sleep(60)
                    else:
                        raise
            self._validate_embeddings(embs, expected_count=len(batch))
            all_embeddings.extend(embs)
            np.save(cache_file, np.asarray(all_embeddings, dtype="float32"))
            print(f"  {len(all_embeddings)}/{len(texts)}")
            time.sleep(pause_s)

        self._validate_embeddings(all_embeddings, expected_count=len(texts))
        embs_np = np.asarray(all_embeddings, dtype="float32")
        faiss.normalize_L2(embs_np)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embs_np)
        self.metadata = documents
        self.save()
        cache_file.unlink(missing_ok=True)
        print(f"Index créé avec succès ({self.index.ntotal} vecteurs).")

    def save(self) -> None:
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(METADATA_FILE, "w", encoding="utf-8") as output:
            json.dump(self.metadata, output, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not FAISS_INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError("Index ou métadonnées absents. Lancez indexer.py d'abord.")
        self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        if self.index.d != self.dimension:
            raise ValueError(
                f"Dimension incompatible : {self.index.d} vs EMBEDDING_DIM={self.dimension}. Réindexez !"
            )
        with open(METADATA_FILE, "r", encoding="utf-8") as source:
            self.metadata = json.load(source)

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Recherche sémantique en instrumentant uniquement les paramètres non sensibles."""
        with span("rag.faiss_search", top_k=top_k) as trace:
            results = self._search_uninstrumented(query, top_k)
            if trace is not None and hasattr(trace, "set_attribute"):
                trace.set_attribute("result_count", len(results))
            return results

    def _search_uninstrumented(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.index is None:
            self.load()
        query_embeddings = self.client.get_embeddings([query], task_type="retrieval_query")
        self._validate_embeddings(query_embeddings, expected_count=1)
        query_vector = np.asarray([query_embeddings[0]], dtype="float32")
        faiss.normalize_L2(query_vector)
        _, indices = self.index.search(query_vector, top_k)
        return [
            self.metadata[idx]
            for idx in indices[0]
            if idx != -1 and idx < len(self.metadata)
        ]
