"""Tests de résilience hors ligne, sans clé/API ni appel réseau."""
from __future__ import annotations
import importlib.util
import json
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_ragas.py"


def load_module(monkeypatch):
    rag = types.ModuleType("rag"); rag.__path__ = []
    ssl = types.ModuleType("rag.ssl_bypass"); ssl.disable_ssl_verification = lambda: None
    config = types.ModuleType("rag.config"); config.PROJECT_ROOT = SCRIPT.parent
    config.MISTRAL_API_KEY = "mock-key"; config.MODEL_NAME = "mistral-small-latest"
    config.EMBEDDING_MODEL = "mistral-embed"; config.SEARCH_K = 5
    vector = types.ModuleType("rag.vector_store"); vector.VectorStoreManager = lambda **kw: None
    llm = types.ModuleType("llm"); llm.__path__ = []
    client = types.ModuleType("llm.mistral_client"); client.MistralClient = lambda **kw: None
    ragas = types.ModuleType("ragas"); ragas.evaluate = lambda **kw: []
    for name, module in {"rag": rag, "rag.ssl_bypass": ssl, "rag.config": config,
                         "rag.vector_store": vector, "llm": llm,
                         "llm.mistral_client": client, "ragas": ragas}.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("evaluate_ragas_resilience_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ev(monkeypatch):
    return load_module(monkeypatch)


def test_throttle_uses_monotonic_and_sleeps(monkeypatch, ev):
    clock = iter([10.0, 11.0, 22.0])
    sleeps = []
    monkeypatch.setattr(ev.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(ev.time, "sleep", sleeps.append)
    ev._last_call_time = None
    ev._throttle(12)
    ev._throttle(12)
    assert sleeps == [11.0]


def test_retry_429_then_success_and_503_then_success(monkeypatch, ev):
    monkeypatch.setattr(ev, "RAGAS_MAX_RETRIES", 3)
    monkeypatch.setattr(ev, "BACKOFF_S", [30, 60, 120, 180, 300, 300])
    monkeypatch.setattr(ev, "_throttle", lambda: None)
    monkeypatch.setattr(ev.random, "uniform", lambda a, b: 0)
    sleeps = []; monkeypatch.setattr(ev.time, "sleep", sleeps.append)
    class Client:
        def __init__(self, errors): self.errors = list(errors)
        def generate(self, **kwargs):
            if self.errors: raise RuntimeError(self.errors.pop(0))
            return "ok"
    assert ev._generate_with_retry(Client(["Status 429 rate_limited"]), "p") == "ok"
    assert ev._generate_with_retry(Client(["Status 503 out_of_capacity"]), "p") == "ok"
    assert sleeps == [30, 30]


def test_non_retryable_error_is_immediate(monkeypatch, ev):
    monkeypatch.setattr(ev, "_throttle", lambda: None)
    calls = []
    class Client:
        def generate(self, **kwargs):
            calls.append(1); raise RuntimeError("Status 400 bad request")
    with pytest.raises(RuntimeError, match="Status 400"):
        ev._generate_with_retry(Client(), "p")
    assert len(calls) == 1


def test_jsonl_append_and_resume_skips_completed_question(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path)
    ev.questions_df = pd.DataFrame({"question": ["already", "new"]})
    jsonl = tmp_path / "data/evaluation/answers_cache.jsonl"
    ev._append_jsonl_response(jsonl, "already", {"answer": "cached", "contexts": ["ctx"]})
    called = []
    class System:
        def __init__(self, **kwargs): pass
        def query(self, question):
            called.append(question)
            return {"answer": "answer", "contexts": ["context"]}
    monkeypatch.setattr(ev, "RAGSystem", System)
    monkeypatch.setattr(ev, "_parse_args", lambda argv=None: types.SimpleNamespace(
        limit=None, categories=None, sleep=0, top_k=3, max_chunk_chars=800,
        max_workers=1, ragas_timeout=300, ragas_retries=10, min_interval=0))
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, False))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["metric"])
    class Scores:
        def to_pandas(self): return pd.DataFrame([{"faithfulness": .9}, {"faithfulness": .8}])
        def __str__(self): return "mock scores"
    monkeypatch.setattr(ev, "evaluate", lambda **kwargs: Scores())
    ev.main([])
    assert called == ["new"]
    records = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [r["question"] for r in records] == ["already", "new"]
    assert ev._load_jsonl_response_cache(jsonl)["new"]["answer"] == "answer"
