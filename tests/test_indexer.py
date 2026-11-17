import json
from pathlib import Path

import pytest
from src.ingestion import indexer


def test_indexer_success_and_offline_manager(monkeypatch,tmp_path):
    data=tmp_path/"processed"; data.mkdir(); (data/"documents.json").write_text('[{"content":"x"}]')
    monkeypatch.setattr(indexer,"DATA_PROCESSED_DIR",data)
    seen=[]
    class Manager:
        def create_index_from_documents(self,docs): seen.extend(docs)
    monkeypatch.setattr(indexer,"VectorStoreManager",Manager)
    indexer.run_indexing()
    assert len(seen) == 1
    assert seen[0]["text"] == "x"
    assert "chunk_index" in seen[0]
    assert "text" in seen[0] and seen[0]["text"] == "x"

def test_missing_input_is_reported(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(indexer,"DATA_PROCESSED_DIR",tmp_path)
    with pytest.raises(FileNotFoundError, match=r"non trouvé\. Exécutez d'abord data_loader\.py"):
        indexer.run_indexing()

def test_indexer_passes_documents_to_store(monkeypatch,tmp_path):
    target=tmp_path/"documents.json"; target.write_text('[{"text":"hello"}]')
    monkeypatch.setattr(indexer,"DATA_PROCESSED_DIR",tmp_path)
    got=[]
    class Store:
        def create_index_from_documents(self, docs): got.extend(docs)
    monkeypatch.setattr(indexer,"VectorStoreManager",Store)
    indexer.run_indexing()
    assert len(got) == 1
    assert got[0]["text"] == "hello"
    assert got[0]["chunk_index"] == 0
    assert got[0]["metadata"]["chunk_index"] == 0
