from pathlib import Path
import numpy as np
import pytest
from src.rag.vector_store import VectorStoreManager


def test_batch_index_cache_save_reload_and_search(tmp_path, monkeypatch, offline_embeddings):
    import src.rag.vector_store as module
    index_file=tmp_path/"index.bin"; meta_file=tmp_path/"meta.json"
    monkeypatch.setattr(module,"FAISS_INDEX_FILE",index_file)
    monkeypatch.setattr(module,"METADATA_FILE",meta_file)
    manager=VectorStoreManager.__new__(VectorStoreManager)
    manager.client=type("Client",(),{"get_embeddings":staticmethod(offline_embeddings)})()
    manager.dimension=3072; manager.index=None; manager.metadata=[]
    sleeps=[]; monkeypatch.setattr(module.time,"sleep",lambda n:sleeps.append(n))
    docs=[{"text":"alpha","content":"alpha"},{"content":"beta"}]
    manager.create_index_from_documents(docs)
    assert manager.index.ntotal==2 and manager.metadata==docs
    assert not Path(str(index_file)+".partial.npy").exists()
    assert sleeps==[20]
    restored=VectorStoreManager.__new__(VectorStoreManager); restored.dimension=3072; restored.client=manager.client; restored.index=None; restored.metadata=[]
    restored.load()
    assert len(restored.search("alpha",top_k=1))==1

def test_partial_cache_resume_and_429_retry(tmp_path, monkeypatch):
    import src.rag.vector_store as module
    monkeypatch.setattr(module,"FAISS_INDEX_FILE",tmp_path/"index.bin"); monkeypatch.setattr(module,"METADATA_FILE",tmp_path/"meta.json")
    manager=VectorStoreManager.__new__(VectorStoreManager); manager.dimension=3072; manager.index=None; manager.metadata=[]
    cache=Path(str(module.FAISS_INDEX_FILE)+".partial.npy"); np.save(cache,np.ones((1,3072),dtype="float32"))
    calls=[]
    def embed(texts,**kwargs):
        calls.append(texts)
        if len(calls)==1: raise module.errors.ClientError("quota",code=429)
        return np.ones((len(texts),3072),dtype="float32").tolist()
    manager.client=type("Client",(),{"get_embeddings":staticmethod(embed)})()
    waits=[]; monkeypatch.setattr(module.time,"sleep",lambda n:waits.append(n))
    manager.create_index_from_documents([{"content":"a"},{"content":"b"}])
    assert calls==[["b"],["b"]] and 60 in waits and not cache.exists()

def test_embedding_failure_propagates_other_than_429(tmp_path,monkeypatch):
    import src.rag.vector_store as module
    monkeypatch.setattr(module,"FAISS_INDEX_FILE",tmp_path/"i"); monkeypatch.setattr(module,"METADATA_FILE",tmp_path/"m")
    m=VectorStoreManager.__new__(VectorStoreManager); m.dimension=3072; m.index=None; m.metadata=[]
    m.client=type("Client",(),{"get_embeddings":lambda self,*a,**k:(_ for _ in ()).throw(RuntimeError("bad"))})()
    monkeypatch.setattr(module.time,"sleep",lambda *_:None)
    with pytest.raises(RuntimeError,match="bad"): m.create_index_from_documents([{"text":"x"}])

def test_load_rejects_wrong_dimension(tmp_path,monkeypatch):
    import src.rag.vector_store as module
    monkeypatch.setattr(module,"FAISS_INDEX_FILE",tmp_path/"i"); monkeypatch.setattr(module,"METADATA_FILE",tmp_path/"m")
    module.faiss.write_index(module.faiss.IndexFlatIP(2),str(module.FAISS_INDEX_FILE)); module.METADATA_FILE.write_text("[]")
    m=VectorStoreManager.__new__(VectorStoreManager); m.dimension=3072; m.index=None; m.metadata=[]
    with pytest.raises(ValueError,match="Dimension incompatible"): m.load()

def test_document_content_field_priority_and_metadata_passthrough(tmp_path,monkeypatch,offline_embeddings):
    import src.rag.vector_store as module
    monkeypatch.setattr(module,"FAISS_INDEX_FILE",tmp_path/"index"); monkeypatch.setattr(module,"METADATA_FILE",tmp_path/"metadata")
    m=VectorStoreManager.__new__(VectorStoreManager); m.dimension=3072; m.index=None; m.metadata=[]
    m.client=type("Client",(),{"get_embeddings":staticmethod(offline_embeddings)})()
    monkeypatch.setattr(module.time,"sleep",lambda *_:None)
    docs=[{"content":"content text","metadata":{"source":"x"}}]
    m.create_index_from_documents(docs)
    assert m.metadata[0]["metadata"]["source"]=="x"
