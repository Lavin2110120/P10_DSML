"""Gestion robuste de l'index FAISS et des chunks documentaires.

Version de diagnostic : chaque étape importante est imprimée et journalisée.
Aucune méthode __getattr__/__getattribute__ personnalisée n'est utilisée.
"""
from __future__ import annotations

import logging
import os
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _debug(message: str) -> None:
    print(f"[VECTOR_STORE] {message}", flush=True)
    logger.info(message)


class VectorStoreManager:
    """Charge et interroge un index FAISS sans mécanisme de récursion implicite."""

    def __init__(self) -> None:
        _debug("__init__: début")
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[Dict[str, Any]] = []
        self.mistral_client: Optional[MistralClient] = None
        try:
            _debug(f"__init__: FAISS_INDEX_FILE={FAISS_INDEX_FILE!r}")
            _debug(f"__init__: DOCUMENT_CHUNKS_FILE={DOCUMENT_CHUNKS_FILE!r}")
            if MISTRAL_API_KEY:
                _debug("__init__: création du client Mistral")
                self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
                _debug("__init__: client Mistral créé")
            else:
                _debug("__init__: clé Mistral absente; recherche API désactivée")
            self._load_index_and_chunks()
            _debug("__init__: fin")
        except Exception:
            print("[VECTOR_STORE] __init__: échec inattendu", flush=True)
            traceback.print_exc()
            raise

    def _load_index_and_chunks(self) -> None:
        """Charge séparément FAISS puis le pickle et valide leur cohérence."""
        index_path = Path(FAISS_INDEX_FILE)
        chunks_path = Path(DOCUMENT_CHUNKS_FILE)
        _debug(f"load: index existe={index_path.is_file()} ({index_path})")
        _debug(f"load: pickle existe={chunks_path.is_file()} ({chunks_path})")
        if not index_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(
                f"Artefacts absents: index={index_path.is_file()}, chunks={chunks_path.is_file()}"
            )

        try:
            _debug("load: avant faiss.read_index")
            loaded_index = faiss.read_index(str(index_path))
            _debug(f"load: FAISS chargé, type={type(loaded_index).__name__}, ntotal={loaded_index.ntotal}, d={loaded_index.d}")
        except Exception as exc:
            raise RuntimeError(f"Échec de lecture FAISS {index_path}: {exc}") from exc

        try:
            _debug("load: avant open/pickle.load")
            with chunks_path.open("rb") as handle:
                loaded_chunks = pickle.load(handle)
            _debug(f"load: pickle chargé, type={type(loaded_chunks).__name__}, longueur={len(loaded_chunks)}")
        except Exception as exc:
            raise RuntimeError(f"Échec de lecture pickle {chunks_path}: {exc}") from exc

        if not isinstance(loaded_chunks, list):
            raise TypeError(f"Le pickle doit contenir une liste, pas {type(loaded_chunks).__name__}")
        for position, chunk in enumerate(loaded_chunks[:5]):
            if not isinstance(chunk, dict) or not {"text", "metadata"}.issubset(chunk):
                raise ValueError(f"Chunk invalide à la position {position}")
        if loaded_index.ntotal != len(loaded_chunks):
            raise ValueError(f"Incohérence: FAISS={loaded_index.ntotal} vecteurs, pickle={len(loaded_chunks)} chunks")

        self.index = loaded_index
        self.document_chunks = loaded_chunks
        _debug("load: validation index/pickle réussie")

    def _split_documents_to_chunks(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        _debug(f"split: {len(documents)} documents")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            length_function=len, add_start_index=True,
        )
        all_chunks: List[Dict[str, Any]] = []
        for doc_number, doc in enumerate(documents):
            langchain_doc = Document(page_content=doc["page_content"], metadata=doc["metadata"])
            for chunk_number, chunk in enumerate(splitter.split_documents([langchain_doc])):
                all_chunks.append({
                    "id": f"{doc_number}_{chunk_number}", "text": chunk.page_content,
                    "metadata": {**chunk.metadata, "chunk_id_in_doc": chunk_number,
                                  "start_index": chunk.metadata.get("start_index", -1)},
                })
        _debug(f"split: {len(all_chunks)} chunks créés")
        return all_chunks

    def _generate_embeddings(self, chunks: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        if not self.mistral_client or not chunks:
            return None
        _debug(f"embeddings: début ({len(chunks)} chunks)")
        vectors: List[Any] = []
        for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL, input=[item["text"] for item in batch]
                )
                vectors.extend(item.embedding for item in response.data)
            except MistralAPIException:
                logger.exception("Erreur API Mistral dans le lot %s", start)
                raise
        result = np.asarray(vectors, dtype="float32")
        _debug(f"embeddings: fin shape={result.shape}")
        return result

    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        chunks = self._split_documents_to_chunks(documents)
        embeddings = self._generate_embeddings(chunks)
        if embeddings is None or len(embeddings) != len(chunks):
            raise RuntimeError("Embeddings absents ou nombre différent du nombre de chunks")
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.document_chunks = chunks
        self._save_index_and_chunks()

    def _save_index_and_chunks(self) -> None:
        if self.index is None or not self.document_chunks:
            raise RuntimeError("Impossible de sauvegarder un index vide")
        Path(FAISS_INDEX_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(DOCUMENT_CHUNKS_FILE).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        with open(DOCUMENT_CHUNKS_FILE, "wb") as handle:
            pickle.dump(self.document_chunks, handle)
        _debug("save: index et chunks sauvegardés")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, Any]]:
        if self.index is None or not self.document_chunks or not self.mistral_client:
            _debug("search: impossible (index/chunks/client indisponible)")
            return []
        try:
            _debug(f"search: embedding de la requête, k={k}")
            response = self.mistral_client.embeddings(model=EMBEDDING_MODEL, input=[query_text])
            query_embedding = np.asarray([response.data[0].embedding], dtype="float32")
            faiss.normalize_L2(query_embedding)
            scores, indices = self.index.search(query_embedding, min(k * 3 if min_score is not None else k, self.index.ntotal))
            results: List[Dict[str, Any]] = []
            for rank, idx in enumerate(indices[0]):
                if 0 <= idx < len(self.document_chunks):
                    raw = float(scores[0][rank])
                    if min_score is None or raw >= min_score:
                        chunk = self.document_chunks[idx]
                        results.append({"score": raw * 100, "raw_score": raw,
                                        "text": chunk["text"], "metadata": chunk["metadata"]})
            return results[:k]
        except Exception:
            logger.exception("search: échec")
            return []
