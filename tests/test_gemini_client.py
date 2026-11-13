from __future__ import annotations
from types import SimpleNamespace
import pytest
import src.llm.gemini_client as module
from src.llm.gemini_client import GeminiClient
import sys
import types
import src.llm.gemini_client as gemini


def make_client():
    client=GeminiClient.__new__(GeminiClient); client.api_key="key"; client.model_name="gemini-test"; client.embedding_model_name="text-embedding-001"; client._backend="google-genai"
    return client

def test_generate_and_stream_use_mocked_google_sdk():
    client=make_client(); calls=[]
    class Models:
        def generate_content(self,**kw): calls.append(kw); return SimpleNamespace(text="answer")
        def generate_content_stream(self,**kw): return iter([SimpleNamespace(text="one"),SimpleNamespace(text=""),SimpleNamespace(text="two")])
    client._client=SimpleNamespace(models=Models())
    assert client.generate_content("question","system",0.1)=="answer"
    assert calls[0]["contents"]=="question" and calls[0]["config"]["system_instruction"]=="system"
    assert list(client.stream_content("q"))==["one","two"]

def test_embedding_batches_normalizes_and_extracts_vectors(monkeypatch):
    client=make_client(); calls=[]
    typesmod=SimpleNamespace(EmbedContentConfig=lambda **kw: SimpleNamespace(**kw))
    import sys, types
    root = types.ModuleType("google"); root.__path__ = []
    genai = types.ModuleType("google.genai"); genai.__path__ = []; genai.types = typesmod
    root.genai = genai
    monkeypatch.setitem(sys.modules, "google", root)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules,"google.genai.types",typesmod)
    class Models:
        def embed_content(self,**kw):
            calls.append(kw)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[1,2]) for _ in kw["contents"]])
    client._client=SimpleNamespace(models=Models())
    got=client.get_embeddings([" a ",None,"c"],batch_size=2,task_type="retrieval-document",output_dimensionality=2)
    assert got==[[1.0,2.0]]*3 and len(calls)==2
    assert calls[0]["config"].task_type=="RETRIEVAL_DOCUMENT"

def test_empty_inputs_and_helper_validation():
    assert module._sanitize_text(None)==""
    assert module._normalize_task_type("semantic_similarity")=="SEMANTIC_SIMILARITY"
    assert module._extract_embedding_vector([1,2])==[1.0,2.0]
    with pytest.raises(TypeError): module._extract_embedding_vector(object())
    client=make_client(); client._client=SimpleNamespace()
    assert client.get_embeddings([])==[]

def test_constructor_uses_mocked_google_genai_client(monkeypatch):
    import sys, types
    calls=[]
    mock=types.ModuleType("google.genai")
    mock.Client=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(models=None)
    mock.Client=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(models=None)
    root = types.ModuleType("google"); root.__path__ = []; root.genai = mock
    monkeypatch.setitem(sys.modules, "google", root)
    monkeypatch.setitem(sys.modules,"google.genai",mock)
    import google
    monkeypatch.setattr(google,"genai",mock,raising=False)
    monkeypatch.setattr(module.config,"GEMINI_API_KEY","key-from-config",raising=False)
    client=GeminiClient(model_name="custom")
    assert client._backend=="google-genai" and client.model_name=="custom"
    assert calls==[{"api_key":"key-from-config"}]

def test_constructor_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(module.config,"GEMINI_API_KEY",None,raising=False)
    monkeypatch.setattr(module.config,"GOOGLE_API_KEY",None,raising=False)
    with pytest.raises(ValueError,match="GEMINI_API_KEY"): GeminiClient(api_key="")

def test_legacy_backend_generation_and_embedding(monkeypatch):
    client=make_client(); client._backend="google-generativeai"
    seen=[]
    class LegacyModel:
        def __init__(self,name,**kw): seen.append((name,kw))
        def generate_content(self,prompt,**kw): return SimpleNamespace(text="legacy answer")
    client._client=SimpleNamespace(GenerativeModel=LegacyModel,embed_content=lambda **kw:{"embedding":[[3,4],[5,6]]})
    assert client.generate_content("q")=="legacy answer"
    assert client.get_embeddings(["a","b"],batch_size=2,task_type="retrieval_query")==[[3.0,4.0],[5.0,6.0]]
    assert seen[0][0]=="gemini-test"
"""Couverture hors ligne des deux backends Gemini."""



def client(backend="google-genai"):
    obj = gemini.GeminiClient.__new__(gemini.GeminiClient)
    obj.api_key = "offline-key"
    obj.model_name = "generation-model"
    obj.embedding_model_name = "embed-model"
    obj._backend = backend
    return obj


def google_modules(monkeypatch):
    root = types.ModuleType("google")
    root.__path__ = []
    genai = types.ModuleType("google.genai")
    genai.__path__ = []
    typ = types.ModuleType("google.genai.types")
    typ.EmbedContentConfig = lambda **kw: SimpleNamespace(**kw)
    root.genai = genai
    genai.types = typ
    monkeypatch.setitem(sys.modules, "google", root)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", typ)


def test_helpers_and_google_client_constructor(monkeypatch):
    assert gemini._sanitize_text(None) == ""
    assert gemini._sanitize_text(23) == "23"
    assert gemini._sanitize_text("  hi  ") == "hi"
    assert gemini._normalize_task_type(None) is None
    assert gemini._normalize_task_type("retrieval_doc") == "RETRIEVAL_DOCUMENT"
    assert gemini._normalize_task_type("custom-task") == "CUSTOM-TASK"
    assert gemini._extract_embedding_vector(SimpleNamespace(values=SimpleNamespace(to_py=lambda: [2, 3]))) == [2., 3.]
    class EmbeddingDict(dict):
        def __getattribute__(self, name):
            if name == "values": raise AttributeError(name)
            return super().__getattribute__(name)
    assert gemini._extract_embedding_vector(EmbeddingDict(embedding=[4, 5])) == [4., 5.]
    with pytest.raises(TypeError):
        gemini._extract_embedding_vector("bad")
    google_modules(monkeypatch)
    import google
    made = []
    google.genai.Client = lambda **kw: made.append(kw) or SimpleNamespace(models=None)
    monkeypatch.setattr(gemini.config, "GEMINI_API_KEY", None, raising=False)
    monkeypatch.setattr(gemini.config, "GOOGLE_API_KEY", "cfg-key", raising=False)
    obj = gemini.GeminiClient()
    assert obj._backend == "google-genai" and made == [{"api_key": "cfg-key"}]
    assert obj.model_name and obj.embedding_model_name
    monkeypatch.setattr(gemini.config, "GEMINI_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "GENERATION_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "GEMINI_EMBEDDING_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "EMBEDDING_MODEL", None, raising=False)
    defaults = gemini.GeminiClient(api_key="direct")
    assert defaults.model_name == "gemini-1.5-flash" and defaults.embedding_model_name == "text-embedding-004"


def test_google_generation_stream_and_batch_embeddings(monkeypatch):
    google_modules(monkeypatch)
    calls = []
    class Models:
        def generate_content(self, **kw):
            calls.append(kw); return SimpleNamespace(text=None)
        def generate_content_stream(self, **kw):
            calls.append(kw); return [SimpleNamespace(text="a"), SimpleNamespace(text=None)]
        def embed_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=SimpleNamespace(to_py=lambda: [1, 2])) for _ in kw["contents"]])
    obj = client(); obj._client = SimpleNamespace(models=Models())
    assert obj.generate_content("q") == ""
    assert "system_instruction" not in calls[-1]["config"]
    assert list(obj.stream_content("q", "sys")) == ["a"]
    result = obj.get_embeddings([" a ", "", "c"], batch_size=2, task_type="retrieval-query", output_dimensionality=2)
    assert result == [[1., 2.]] * 3
    embeds = [c for c in calls if c.get("model") == "embed-model"]
    assert len(embeds) == 2 and embeds[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert obj.get_embedding("  ") == [1., 2.]
    assert obj.get_embeddings(iter(())) == []


def test_google_embeddings_without_optional_config_and_response_list(monkeypatch):
    google_modules(monkeypatch)
    obj = client(); obj._client = SimpleNamespace(models=SimpleNamespace(
        embed_content=lambda **kw: SimpleNamespace(embeddings=[SimpleNamespace(values=[7, "8"])])))
    assert obj.get_embeddings(["x"]) == [[7., 8.]]


def test_legacy_constructor_generation_stream_and_embed_shapes(monkeypatch):
    google_modules(monkeypatch)
    monkeypatch.setattr(gemini.config, "GEMINI_API_KEY", None, raising=False)
    monkeypatch.setattr(gemini.config, "GOOGLE_API_KEY", None, raising=False)
    with pytest.raises(ValueError):
        gemini.GeminiClient(api_key="")
    # Exercise constructor's google-generativeai fallback via a deliberately
    # absent google.genai import and an in-memory legacy SDK module.
    root = sys.modules["google"]
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai.types", raising=False)
    legacy = types.ModuleType("google.generativeai")
    legacy.configure = lambda **kw: None
    legacy.GenerativeModel = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "google.generativeai", legacy)
    monkeypatch.delattr(root, "genai", raising=False)
    import builtins
    original_import = builtins.__import__
    def import_without_genai(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and "genai" in fromlist:
            raise ImportError("modern SDK missing")
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", import_without_genai)
    constructed = gemini.GeminiClient(api_key="legacy-key")
    assert constructed._backend == "google-generativeai" and constructed._client is legacy
    # Since import machinery sees the legacy child in sys.modules, direct path is deterministic.
    obj = client("google-generativeai")
    events = []
    class Model:
        def __init__(self, name, **kw): events.append((name, kw))
        def generate_content(self, *args, **kw):
            events.append((args, kw))
            if kw.get("stream"):
                return [SimpleNamespace(text="one"), SimpleNamespace(text=None), SimpleNamespace(text="two")]
            return SimpleNamespace(text="legacy")
    replies = iter([{"embedding": [1, 2]}, {"embedding": [[3, 4], [5, 6]]}])
    obj._client = SimpleNamespace(GenerativeModel=Model,
        embed_content=lambda **kw: next(replies))
    assert obj.generate_content("prompt", "system") == "legacy"
    assert events[0][1]["system_instruction"] == "system"
    assert list(obj.stream_content("prompt")) == ["one", "two"]
    assert obj.get_embeddings(["x"]) == [[1., 2.]]
    assert obj.get_embeddings(["a", "b"], batch_size=2, task_type="fact-verification", output_dimensionality=2) == [[3., 4.], [5., 6.]]
    assert events[2][0] == "generation-model"
