from __future__ import annotations
from types import SimpleNamespace
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources
import src.ingestion.indexer as indexer

@pytest.mark.unit
def test_index_documents_loads_builds_and_returns_manager(monkeypatch):
    docs = [{"text": "NBA document"}]
    built = []
    class Manager:
        index = SimpleNamespace(ntotal=1)
        document_chunks = [{"text": "NBA document"}]
        def __init__(self, client=None, **kwargs): self.client = client
        def build_index(self, items): built.append(items)
    class Client:
        def __init__(self, api_key=None, **kwargs): self.api_key = api_key
    monkeypatch.setattr(indexer.config, "MISTRAL_API_KEY", "valid-test-key")
    monkeypatch.setattr(indexer, "MistralClient", Client)
    monkeypatch.setattr(indexer, "load_processed_documents", lambda path: docs if path == "input.json" else [])
    monkeypatch.setattr(indexer, "VectorStoreManager", Manager)
    manager = indexer.index_documents("input.json")
    assert isinstance(manager, Manager) and built == [docs]

@pytest.mark.unit
def test_index_documents_empty_index_is_logged_as_zero(monkeypatch):
    class Manager:
        index = None
        document_chunks = []
        def __init__(self, client=None, **kwargs): self.client = client
        def build_index(self, docs): assert docs == []
    class Client:
        def __init__(self, api_key=None, **kwargs): self.api_key = api_key
    monkeypatch.setattr(indexer.config, "MISTRAL_API_KEY", "valid-test-key")
    monkeypatch.setattr(indexer, "MistralClient", Client)
    monkeypatch.setattr(indexer, "load_processed_documents", lambda path: [])
    monkeypatch.setattr(indexer, "VectorStoreManager", Manager)
    assert indexer.index_documents() .index is None

@pytest.mark.unit
def test_main_forwards_cli_input(monkeypatch):
    called = []
    monkeypatch.setattr(indexer, "index_documents", lambda path: called.append(path))
    monkeypatch.setattr("sys.argv", ["indexer", "--input", "docs.json"])
    indexer.main()
    assert called == ["docs.json"]
