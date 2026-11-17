"""Offline validation checks for the Pydantic data contracts."""
import pytest
from pydantic import ValidationError
from src.models import Document, Chunk, RAGAnswer

def test_document_accepts_historical_text_aliases():
    for key in ("text", "content", "page_content"):
        doc = Document.model_validate({key: " valid ", "metadata": {"source": "x", "page": 1}})
        assert doc.text == "valid" and doc.source == "x" and doc.page == 1

def test_invalid_document_and_chunk_rejected():
    with pytest.raises(ValidationError): Document.model_validate({"text": "  "})
    with pytest.raises(ValidationError): Chunk.model_validate({"text": "ok", "chunk_index": -1})

def test_chunk_alias_index_and_answer_validation():
    chunk=Chunk.model_validate({"content":"evidence", "metadata":{"chunk_index":2}})
    assert chunk.text == "evidence" and chunk.chunk_index == 2
    answer=RAGAnswer.model_validate({"answer":"ok", "sources":[chunk], "confidence":0.7})
    assert answer.answer == "ok"
    with pytest.raises(ValidationError): RAGAnswer.model_validate({"answer":"  ", "sources":[]})
