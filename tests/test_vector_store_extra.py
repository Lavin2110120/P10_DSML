from __future__ import annotations
import pickle
import numpy as np
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources
import src.rag.vector_store as module
from src.rag.vector_store import VectorStoreManager

class Client:
    def __init__(self, vectors=None): self.vectors, self.calls = vectors, []
    def embeddings(self, texts):
        self.calls.append(list(texts))
        return self.vectors(texts) if callable(self.vectors) else self.vectors

def manager(tmp_path, monkeypatch, client=None):
    monkeypatch.setattr(module.config, "FAISS_INDEX_FILE", tmp_path / "index.faiss")
    monkeypatch.setattr(module.config, "DOCUMENT_CHUNKS_FILE", tmp_path / "chunks.pkl")
    monkeypatch.setattr(module.config, "CHUNK_SIZE", 4)
    monkeypatch.setattr(module.config, "CHUNK_OVERLAP", 1)
    return VectorStoreManager(client=client)

@pytest.mark.unit
def test_make_chunks_spans_overlap_and_normalizes_metadata(monkeypatch):
    monkeypatch.setattr(module.config, "CHUNK_SIZE", 4)
    monkeypatch.setattr(module.config, "CHUNK_OVERLAP", 1)
    chunks = VectorStoreManager._make_chunks([{"text": "abcdefgh", "source": "book", "metadata": "legacy"}])
    assert [c["text"] for c in chunks] == ["abcd", "defg", "gh"]
    assert chunks[0]["metadata"]["source"] == "book"
    assert chunks[0]["metadata"]["metadata"] == "legacy"
    assert chunks[1]["metadata"]["start_index"] == 3

@pytest.mark.unit
def test_chunk_input_validation(monkeypatch):
    monkeypatch.setattr(module.config, "CHUNK_SIZE", 3)
    monkeypatch.setattr(module.config, "CHUNK_OVERLAP", 3)
    with pytest.raises(ValueError): VectorStoreManager._make_chunks([])
    monkeypatch.setattr(module.config, "CHUNK_OVERLAP", 0)
    with pytest.raises(TypeError): VectorStoreManager._make_chunks(["no dict"])
    with pytest.raises(TypeError): VectorStoreManager._make_chunks([{"text": 42}])
    assert VectorStoreManager._make_chunks([{"text": "   "}]) == []

@pytest.mark.unit
def test_build_search_save_reload_and_filter(tmp_path, monkeypatch):
    client = Client(lambda texts: [[float(len(text)), 0.0] for text in texts])
    store = manager(tmp_path, monkeypatch, client)
    store.build_index([{"text": "abcdEF", "metadata": {"source": "x"}}])
    assert store.index.ntotal == len(store.document_chunks) == 2
    assert store.index_path.is_file() and store.chunks_path.is_file()
    hits = store.search("abcd", k=10)
    assert len(hits) == 2 and hits[0]["score"] == 0
    assert store.search("abcd", min_score=-1) == []
    assert store.search(" ") == [] and store.search("abcd", k=0) == []
    loaded = manager(tmp_path, monkeypatch, client)
    assert loaded.index.ntotal == 2 and loaded.chunks == store.chunks

@pytest.mark.unit
def test_search_empty_missing_client_bad_dimension_and_empty_vectors(tmp_path, monkeypatch):
    store = manager(tmp_path, monkeypatch)
    assert store.search("query") == []
    store.index = module.faiss.IndexFlatL2(2); store.index.add(np.array([[1, 0]], dtype=np.float32))
    store.document_chunks = [{"text": "hit"}]
    assert store.search("query") == []
    store.mistral_client = Client([])
    assert store.search("query") == []
    store.mistral_client = Client([[1, 2, 3]])
    with pytest.raises(ValueError): store.search("query")

@pytest.mark.unit
def test_empty_build_removes_artifacts_and_invalid_embeddings_fail(tmp_path, monkeypatch):
    store = manager(tmp_path, monkeypatch, Client([[1, 0]]))
    store.index_path.parent.mkdir(parents=True, exist_ok=True); store.index_path.touch(); store.chunks_path.touch()
    store.build_index([])
    assert store.index is None and not store.index_path.exists() and not store.chunks_path.exists()
    store.mistral_client = None
    with pytest.raises(RuntimeError): store.build_index([{"text": "a"}])
    store.mistral_client = Client([[1, 0]])
    monkeypatch.setattr(module.config, "EMBEDDING_BATCH_SIZE", 0)
    with pytest.raises(ValueError): store.build_index([{"text": "a"}])
    monkeypatch.setattr(module.config, "EMBEDDING_BATCH_SIZE", 32)
    monkeypatch.setattr(module.config, "EMBEDDING_DIM", 3, raising=False)
    with pytest.raises(ValueError, match="Dimension inattendue"):
        store.build_index([{"text": "a"}])

@pytest.mark.unit
def test_load_rejects_corrupt_pickle_and_mismatched_chunks(tmp_path, monkeypatch):
    store = manager(tmp_path, monkeypatch)
    store.index_path.parent.mkdir(parents=True, exist_ok=True); store.index_path.touch()
    store.chunks_path.write_bytes(pickle.dumps(["bad chunk"]))
    assert store.load() is False and store.index is None
    with store.chunks_path.open("wb") as stream: pickle.dump([], stream)
    monkeypatch.setattr(module.faiss, "read_index", lambda path: type("I", (), {"ntotal": 1})())
    assert store.load() is False

@pytest.mark.unit
def test_save_requires_index_and_add_documents_alias(tmp_path, monkeypatch):
    store = manager(tmp_path, monkeypatch, Client(lambda texts: [[1., 2.] for _ in texts]))
    with pytest.raises(ValueError): store.save()
    store.add_documents([{"content": "abc"}])
    assert store.index.ntotal == 1
