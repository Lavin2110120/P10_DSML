"""Tests fonctionnels ciblés de VectorStoreManager."""
from __future__ import annotations

import pytest

from src.rag.vector_store import VectorStoreManager


@pytest.mark.unit
def test_split_documents_to_chunks_preserves_content_and_metadata() -> None:
    manager = VectorStoreManager.__new__(VectorStoreManager)
    documents = [
        {
            "page_content": "Le contenu du document à découper.",
            "metadata": {"source": "fixture.txt", "filename": "fixture.txt"},
        }
    ]

    chunks = manager._split_documents_to_chunks(documents)

    assert chunks
    assert chunks[0]["id"] == "0_0"
    assert chunks[0]["text"] == documents[0]["page_content"]
    assert chunks[0]["metadata"]["source"] == "fixture.txt"
    assert chunks[0]["metadata"]["chunk_id_in_doc"] == 0
    assert "start_index" in chunks[0]["metadata"]


@pytest.mark.integration
def test_splitter_handles_multiple_documents() -> None:
    manager = VectorStoreManager.__new__(VectorStoreManager)
    chunks = manager._split_documents_to_chunks(
        [
            {"page_content": "Premier texte.", "metadata": {"source": "a.txt"}},
            {"page_content": "Second texte.", "metadata": {"source": "b.txt"}},
        ]
    )

    assert [chunk["id"] for chunk in chunks] == ["0_0", "1_0"]
    assert [chunk["metadata"]["source"] for chunk in chunks] == ["a.txt", "b.txt"]
