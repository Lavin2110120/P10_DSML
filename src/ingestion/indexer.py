"""Normalisation, validation et indexation FAISS des documents du corpus."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.config import DATA_PROCESSED_DIR
from src.ingestion.data_loader import chunk_documents, normalize_text
from src.models import Chunk, Document
from src.observability import span
from src.rag.vector_store import VectorStoreManager

_LOG = logging.getLogger(__name__)


def _validated_documents(items: Any) -> list[dict[str, Any]]:
    """Valide et normalise les documents JSON, en ignorant les entrées invalides."""
    if not isinstance(items, list):
        raise ValueError("Le corpus de documents doit être une liste JSON")
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        try:
            document = Document.model_validate(item)
            document.text = normalize_text(document.text)
            # Revalidation après normalisation: le texte vide n'est pas indexable.
            document = Document.model_validate(document.model_dump(mode="python", exclude_none=True))
            valid.append(document.model_dump(mode="python", exclude_none=True))
        except Exception as exc:
            _LOG.warning("Document %d ignoré : validation impossible (%s)", index, exc)
    return valid


def run_indexing(input_file: str | Path | None = None) -> None:
    source = Path(input_file) if input_file is not None else Path(DATA_PROCESSED_DIR) / "documents.json"
    if not source.exists():
        raise FileNotFoundError(f"{source} non trouvé. Exécutez d'abord data_loader.py !")
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)

    documents = _validated_documents(raw)
    with span("rag.normalize_and_chunk", document_count=len(documents)):
        raw_chunks = chunk_documents(documents)
        chunks: list[dict[str, Any]] = []
        for index, item in enumerate(raw_chunks):
            try:
                chunks.append(Chunk.model_validate(item).model_dump(mode="python", exclude_none=True))
            except Exception as exc:
                _LOG.warning("Chunk %d ignoré : validation impossible (%s)", index, exc)
    _LOG.info("Normalisation et découpage : %d documents -> %d chunks", len(documents), len(chunks))
    print(f"Normalisation et découpage : {len(documents)} documents -> {len(chunks)} chunks")
    if not chunks:
        raise ValueError("Aucun chunk valide à indexer.")

    with span("rag.index_documents", chunk_count=len(chunks)):
        VectorStoreManager().create_index_from_documents(chunks)
    print("✅ Indexation FAISS avec Gemini terminée !")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_indexing()
