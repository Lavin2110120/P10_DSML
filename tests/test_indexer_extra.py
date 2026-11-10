from __future__ import annotations
import pytest
import rag_test_support  # noqa
import src.ingestion.indexer as mod

@pytest.mark.unit
def test_api_key_missing_placeholder_and_normalized_value(monkeypatch):
    for value in [None,"", "   ","test-key","CHANGEme","your_api_key","your-api-key","your-mistral-api-key","replace-me","replace_me","none","NULL"]:
        monkeypatch.setattr(mod.config,"MISTRAL_API_KEY",value)
        with pytest.raises(RuntimeError,match="MISTRAL_API_KEY"): mod._configured_api_key()
    monkeypatch.setattr(mod.config,"MISTRAL_API_KEY","  actual-secret  ")
    assert mod._configured_api_key()=="actual-secret"

@pytest.mark.unit
def test_index_documents_passes_loaded_docs_and_validated_key(monkeypatch):
    docs=[{"text":"source"}]; events=[]
    monkeypatch.setattr(mod,"_configured_api_key",lambda:"key")
    monkeypatch.setattr(mod,"load_processed_documents",lambda path:events.append(("load",path)) or docs)
    class Client:
        def __init__(self,api_key): events.append(("client",api_key))
    class Store:
        index=type("Idx",(),{"ntotal":3})()
        def __init__(self,client): events.append(("store",client))
        def build_index(self,items): events.append(("build",items))
    monkeypatch.setattr(mod,"MistralClient",Client); monkeypatch.setattr(mod,"VectorStoreManager",Store)
    manager=mod.index_documents("chosen.json")
    assert isinstance(manager,Store)
    assert [e[0] for e in events]==["load","client","store","build"] and events[-1]==("build",docs)

@pytest.mark.unit
def test_main_no_argument_forwards_none(monkeypatch):
    seen=[]; monkeypatch.setattr(mod,"index_documents",lambda path:seen.append(path) or "done")
    monkeypatch.setattr("sys.argv",["indexer"])
    assert mod.main()=="done" and seen==[None]
