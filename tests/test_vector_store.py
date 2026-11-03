"""Tests fonctionnels ciblés de VectorStoreManager."""
from __future__ import annotations

import pytest

from src.rag.vector_store import VectorStoreManager


@pytest.mark.unit
def test_make_chunks_preserves_content_and_metadata() -> None:
    # _make_chunks est une staticmethod : on peut l'appeler sur la classe
    # ou sur une instance "vide" créée avec __new__
    documents = [
        {
            "text": "Le contenu du document à découper.",
            "source": "fixture.txt",
            "page": 1,
            "category": None,
        }
    ]

    chunks = VectorStoreManager._make_chunks(documents)

    assert chunks
    assert chunks[0]["id"] == "0_0"
    assert chunks[0]["text"] == "Le contenu du document à découper."
    # Selon la structure réelle de _make_chunks :
    assert chunks[0]["metadata"]["source"] == "fixture.txt"
    assert chunks[0]["metadata"]["chunk_id_in_doc"] == 0
    assert "start_index" in chunks[0]["metadata"]


@pytest.mark.integration
def test_make_chunks_handles_multiple_documents() -> None:
    chunks = VectorStoreManager._make_chunks(
        [
            {"text": "Premier texte.", "source": "a.txt", "page": 1},
            {"text": "Second texte.", "source": "b.txt", "page": 1},
        ]
    )

    assert [chunk["id"] for chunk in chunks] == ["0_0", "1_0"]
    assert [chunk["metadata"]["source"] for chunk in chunks] == ["a.txt", "b.txt"]