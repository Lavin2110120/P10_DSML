"""Tests du pipeline RAG avec le SDK Gemini simulé (sans trafic réseau)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from src.llm.gemini_client import GeminiClient
from src.rag.vector_store import VectorStoreManager


@pytest.mark.unit
def test_gemini_client_generates_answer_with_mocked_sdk() -> None:
    expected = "Réponse simulée"
    calls = []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=expected)

    client = GeminiClient.__new__(GeminiClient)
    client.model_name = "gemini-test"
    client._backend = "google-genai"
    client._client = SimpleNamespace(models=FakeModels())

    assert client.generate_content(
        "Question ?", system_prompt="Réponds brièvement.", temperature=0.1
    ) == expected
    assert calls == [{
        "model": "gemini-test",
        "contents": "Question ?",
        "config": {"temperature": 0.1, "system_instruction": "Réponds brièvement."},
    }]


@pytest.mark.unit
def test_rag_components_import_from_src_packages() -> None:
    """Vérifie que les composants Gemini et recherche sont importables."""
    assert GeminiClient.__name__ == "GeminiClient"
    assert VectorStoreManager.__name__ == "VectorStoreManager"

# Additional branch coverage for the Gemini client, kept in this requested test file.
import sys
import types
import src.llm.gemini_client as gemini


def _gemini_client(backend="google-genai"):
    obj = gemini.GeminiClient.__new__(gemini.GeminiClient)
    obj.api_key = "offline-key"
    obj.model_name = "generation-model"
    obj.embedding_model_name = "embed-model"
    obj._backend = backend
    return obj


def _install_google_modules(monkeypatch):
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


def test_gemini_helpers_and_constructor_paths(monkeypatch):
    assert gemini._sanitize_text(None) == ""
    assert gemini._sanitize_text(23) == "23"
    assert gemini._sanitize_text("  hi  ") == "hi"
    assert gemini._normalize_task_type(None) is None
    assert gemini._normalize_task_type("retrieval_doc") == "RETRIEVAL_DOCUMENT"
    assert gemini._normalize_task_type("custom-task") == "CUSTOM-TASK"
    assert gemini._extract_embedding_vector(SimpleNamespace(values=SimpleNamespace(to_py=lambda: [2, 3]))) == [2.0, 3.0]
    class EmbedDict(dict):
        def __getattribute__(self, name):
            if name == "values":
                raise AttributeError(name)
            return super().__getattribute__(name)
    assert gemini._extract_embedding_vector(EmbedDict(embedding=[4, 5])) == [4.0, 5.0]
    with pytest.raises(TypeError):
        gemini._extract_embedding_vector("invalid")

    _install_google_modules(monkeypatch)
    import google
    made = []
    google.genai.Client = lambda **kw: made.append(kw) or SimpleNamespace(models=None)
    monkeypatch.setattr(gemini.config, "GEMINI_API_KEY", None, raising=False)
    monkeypatch.setattr(gemini.config, "GOOGLE_API_KEY", "cfg-key", raising=False)
    client = gemini.GeminiClient()
    assert client._backend == "google-genai" and made == [{"api_key": "cfg-key"}]
    monkeypatch.setattr(gemini.config, "GEMINI_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "GENERATION_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "GEMINI_EMBEDDING_MODEL", None, raising=False)
    monkeypatch.setattr(gemini.config, "EMBEDDING_MODEL", None, raising=False)
    defaults = gemini.GeminiClient(api_key="direct")
    assert defaults.model_name == "gemini-1.5-flash"
    assert defaults.embedding_model_name == "text-embedding-004"


def test_gemini_google_stream_and_embeddings(monkeypatch):
    _install_google_modules(monkeypatch)
    calls = []
    class Models:
        def generate_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(text=None)
        def generate_content_stream(self, **kw):
            calls.append(kw)
            return [SimpleNamespace(text="a"), SimpleNamespace(text=None)]
        def embed_content(self, **kw):
            calls.append(kw)
            return SimpleNamespace(embeddings=[SimpleNamespace(values=SimpleNamespace(to_py=lambda: [1, 2])) for _ in kw["contents"]])
    client = _gemini_client()
    client._client = SimpleNamespace(models=Models())
    assert client.generate_content("q") == ""
    assert "system_instruction" not in calls[-1]["config"]
    assert list(client.stream_content("q", "system")) == ["a"]
    assert client.get_embeddings([" a ", "", "c"], batch_size=2, task_type="retrieval-query", output_dimensionality=2) == [[1.0, 2.0]] * 3
    embedding_calls = [c for c in calls if c.get("model") == "embed-model"]
    assert len(embedding_calls) == 2
    assert embedding_calls[0]["config"].task_type == "RETRIEVAL_QUERY"
    assert client.get_embedding("  ") == [1.0, 2.0]
    assert client.get_embeddings(iter(())) == []


def test_gemini_google_embeddings_without_config_and_legacy_backend(monkeypatch):
    _install_google_modules(monkeypatch)
    client = _gemini_client()
    client._client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kw: SimpleNamespace(embeddings=[SimpleNamespace(values=[7, "8"])])))
    assert client.get_embeddings(["x"]) == [[7.0, 8.0]]

    import builtins
    root = sys.modules["google"]
    monkeypatch.delitem(sys.modules, "google.genai", raising=False)
    monkeypatch.delitem(sys.modules, "google.genai.types", raising=False)
    monkeypatch.delattr(root, "genai", raising=False)
    legacy = types.ModuleType("google.generativeai")
    legacy.configure = lambda **kw: None
    legacy.GenerativeModel = lambda *a, **kw: None
    monkeypatch.setitem(sys.modules, "google.generativeai", legacy)
    original_import = builtins.__import__
    def without_genai(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and "genai" in fromlist:
            raise ImportError("modern SDK unavailable")
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", without_genai)
    constructed = gemini.GeminiClient(api_key="legacy-key")
    assert constructed._backend == "google-generativeai"

    legacy_client = _gemini_client("google-generativeai")
    events = []
    class LegacyModel:
        def __init__(self, name, **kw): events.append((name, kw))
        def generate_content(self, *args, **kw):
            events.append((args, kw))
            if kw.get("stream"):
                return [SimpleNamespace(text="one"), SimpleNamespace(text=None), SimpleNamespace(text="two")]
            return SimpleNamespace(text="legacy")
    replies = iter([{"embedding": [1, 2]}, {"embedding": [[3, 4], [5, 6]]}])
    legacy_client._client = SimpleNamespace(GenerativeModel=LegacyModel, embed_content=lambda **kw: next(replies))
    assert legacy_client.generate_content("prompt", "system") == "legacy"
    assert list(legacy_client.stream_content("prompt")) == ["one", "two"]
    assert legacy_client.get_embeddings(["x"]) == [[1.0, 2.0]]
    assert legacy_client.get_embeddings(["a", "b"], batch_size=2, task_type="fact-verification", output_dimensionality=2) == [[3.0, 4.0], [5.0, 6.0]]
    assert events[0][0] == "generation-model"
