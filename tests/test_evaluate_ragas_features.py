"""Tests unitaires hors ligne des nouveautés evaluate_ragas (aucun appel Mistral/RAGAS)."""
from __future__ import annotations
import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "src" / "evaluation" / "evaluate_ragas.py"


def _load_module(monkeypatch):
    rag = types.ModuleType("rag"); rag.__path__ = []
    ssl = types.ModuleType("rag.ssl_bypass"); ssl.disable_ssl_verification = lambda: None
    config = types.ModuleType("rag.config"); config.PROJECT_ROOT = PROJECT_ROOT
    config.MISTRAL_API_KEY = "mock-key"; config.MODEL_NAME = "mock-model"
    config.EMBEDDING_MODEL = "mock-embed"; config.SEARCH_K = 5
    vector = types.ModuleType("rag.vector_store"); vector.VectorStoreManager = lambda **kw: None
    llm_pkg = types.ModuleType("llm"); llm_pkg.__path__ = []
    client = types.ModuleType("llm.mistral_client"); client.MistralClient = lambda **kw: None
    ragas = types.ModuleType("ragas"); ragas.evaluate = lambda **kw: []
    modules = {"rag": rag, "rag.ssl_bypass": ssl, "rag.config": config,
               "rag.vector_store": vector, "llm": llm_pkg,
               "llm.mistral_client": client, "ragas": ragas}
    for name, value in modules.items(): monkeypatch.setitem(sys.modules, name, value)
    spec = importlib.util.spec_from_file_location("evaluate_ragas_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ev(monkeypatch):
    return _load_module(monkeypatch)


def test_cli_limit_and_category_variants(ev):
    args = ev._parse_args(["--limit", "3", "--categories", "simple,complexe", "bruitée"])
    assert args.limit == 3
    assert args.categories == ["simple", "complexe", "bruitée"]
    assert (args.max_workers, args.ragas_timeout, args.ragas_retries) == (1, 300, 10)
    assert args.sleep == 0.0
    assert (args.top_k, args.max_chunk_chars) == (3, 800)
    custom = ev._parse_args(["--max-workers", "4", "--ragas-timeout", "99", "--ragas-retries", "8", "--sleep", "0.25", "--top-k", "6", "--max-chunk-chars", "1200"])
    assert (custom.max_workers, custom.ragas_timeout, custom.ragas_retries) == (4, 99, 8)
    assert custom.sleep == 0.25 and custom.top_k == 6 and custom.max_chunk_chars == 1200
    with pytest.raises(SystemExit):
        ev._parse_args(["--limit", "-1"])


def test_cache_roundtrip_and_reject_bad_cache(ev, tmp_path):
    path = tmp_path / "cache.json"
    initial = {"question": {"answer": "réponse", "contexts": ["contexte"]},
               "bad": {"answer": "", "contexts": []}}
    ev._save_response_cache(path, initial)
    assert ev._load_response_cache(path) == {"question": initial["question"]}
    path.write_text("not json", encoding="utf-8")
    assert ev._load_response_cache(path) == {}


def test_main_partial_fail_cache_and_category_reports(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path)
    monkeypatch.setattr(ev, "RAGAS_QUESTION_DELAY", 0)
    ev.questions_df = pd.DataFrame({
        "question": ["q1", "q2", "q3"],
        "ground_truth_expected": ["gt1", "gt2", "gt3"],
        "category": ["simple", "complexe", "simple"],
    })
    monkeypatch.setattr(ev, "_parse_args", lambda argv=None: types.SimpleNamespace(
        limit=2, categories=["simple", "complexe"]))
    calls = []
    class System:
        def query(self, question):
            calls.append(question)
            if question == "q2":
                raise RuntimeError("simulated offline failure")
            return {"answer": "answer-" + question, "contexts": ["context-" + question]}
    monkeypatch.setattr(ev, "RAGSystem", System)
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, True))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["mock-metric"])
    scored = []
    class MockScores:
        def to_pandas(self):
            return pd.DataFrame([{"faithfulness": 0.75}])
        def __str__(self):
            return "simulated scores"
    def mock_evaluate(**kwargs):
        scored.extend(kwargs["dataset"])
        return MockScores()
    monkeypatch.setattr(ev, "evaluate", mock_evaluate)

    ev.main([])
    assert calls == ["q1", "q2"]
    assert len(scored) == 1 and scored[0]["user_input"] == "q1"
    cache = ev._load_response_cache(tmp_path / "data/evaluation/cache_reponses.json")
    assert "q1" in cache and "q2" not in cache
    detail = pd.read_csv(tmp_path / "data/evaluation/ragas_detail_par_question.csv")
    aggregate = pd.read_csv(tmp_path / "data/evaluation/ragas_scores_par_categorie.csv")
    assert list(detail["category"]) == ["simple"]
    assert "faithfulness" in detail.columns
    assert aggregate.to_dict("records")[0]["question_count"] == 1
    assert (tmp_path / "data/evaluation/ragas_scores_par_categorie.md").exists()


def test_category_filter_is_applied_before_limit(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path)
    monkeypatch.setattr(ev, "RAGAS_QUESTION_DELAY", 0)
    ev.questions_df = pd.DataFrame({"question": ["a", "b", "c"],
                                    "category": ["simple", "complexe", "simple"]})
    monkeypatch.setattr(ev, "_parse_args", lambda argv=None: types.SimpleNamespace(
        limit=1, categories=["complexe"]))
    called = []
    class System:
        def query(self, question):
            called.append(question)
            return {"answer": "answer", "contexts": ["context"]}
    monkeypatch.setattr(ev, "RAGSystem", System)
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, False))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["metric"])
    monkeypatch.setattr(ev, "evaluate", lambda **kw: pd.DataFrame([{"faithfulness": .5}]))
    ev.main([])
    assert called == ["b"]


def test_question_csv_schema_and_allowed_categories(ev):
    valid = pd.DataFrame({
        "question": ["q"],
        "ground_truth_expected": ["référence"],
        "category": ["hors-périmètre/numérique"],
    })
    assert ev._validate_questions_df(valid).equals(valid)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        ev._validate_questions_df(valid.drop(columns=["ground_truth_expected"]))
    invalid = valid.copy()
    invalid.loc[0, "category"] = "hors_perimetre"
    with pytest.raises(ValueError, match="Catégories inconnues"):
        ev._validate_questions_df(invalid)


def test_updated_real_csv_matches_evaluator_schema(ev):
    csv_path = PROJECT_ROOT / "data" / "evaluation" / "questions_metier.csv"
    if not csv_path.exists():
        pytest.skip("Le CSV métier est absent à la racine du projet")
    frame = pd.read_csv(csv_path)
    checked = ev._validate_questions_df(frame)
    assert len(checked) == 30
    assert set(checked["category"]) == ev.VALID_CATEGORIES


def test_top_k_and_context_chunks_are_truncated_consistently(ev, monkeypatch):
    searched = {}
    prompts = []
    class Store:
        chunks = ["present"]
        def __init__(self, client=None): pass
        def search(self, question, k):
            searched["k"] = k
            return [{"text": "abcdefghij"}, {"text": "0123456789"}]
    class Client:
        def generate(self, prompt, max_tokens):
            prompts.append(prompt)
            return "answer"
    monkeypatch.setattr(ev, "VectorStoreManager", Store)
    monkeypatch.setattr(ev, "MistralClient", Client)
    response = ev.RAGSystem(top_k=4, max_chunk_chars=5).query("q")
    assert searched["k"] == 4 and response["contexts"] == ["abcde", "01234"]
    assert "abcdefghij" not in prompts[0]


def test_sleep_only_between_uncached_generation_calls(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path)
    ev.questions_df = pd.DataFrame({"question": ["cached", "live1", "live2"]})
    class System:
        def __init__(self, **kwargs): pass
        def query(self, question): return {"answer": "a-" + question, "contexts": ["context"]}
    monkeypatch.setattr(ev, "RAGSystem", System)
    monkeypatch.setattr(ev, "_parse_args", lambda argv=None: types.SimpleNamespace(
        limit=None, categories=None, sleep=1.25, top_k=3, max_chunk_chars=800,
        max_workers=2, ragas_timeout=180, ragas_retries=10))
    ev._save_response_cache(tmp_path / "data/evaluation/cache_reponses.json", {
        "cached": {"answer": "from-cache", "contexts": ["cache-context"]}})
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, True))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["metric"])
    sleeps=[]; monkeypatch.setattr(ev.time, "sleep", lambda seconds: sleeps.append(seconds))
    class Scores:
        def to_pandas(self): return pd.DataFrame([{"faithfulness": .9}] * 3)
        def __str__(self): return "scores"
    monkeypatch.setattr(ev, "evaluate", lambda **kw: Scores())
    ev.main([])
    assert sleeps == [1.25]


def test_run_config_passed_and_old_evaluate_supported(ev, monkeypatch, tmp_path):
    monkeypatch.setattr(ev, "project_root", tmp_path)
    ev.questions_df = pd.DataFrame({"question": ["q"]})
    class System:
        def __init__(self, **kwargs): pass
        def query(self, question): return {"answer": "a", "contexts": ["c"]}
    monkeypatch.setattr(ev, "RAGSystem", System)
    monkeypatch.setattr(ev, "_parse_args", lambda argv=None: types.SimpleNamespace(
        limit=None, categories=None, sleep=0, top_k=3, max_chunk_chars=800,
        max_workers=5, ragas_timeout=123, ragas_retries=9))
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, True))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["metric"])
    rc=types.ModuleType("ragas.run_config")
    class RunConfig:
        def __init__(self, max_workers=1, timeout=60, max_retries=3):
            self.max_workers,self.timeout,self.max_retries=max_workers,timeout,max_retries
    rc.RunConfig=RunConfig; monkeypatch.setitem(sys.modules,"ragas.run_config",rc)
    class Scores:
        def to_pandas(self): return pd.DataFrame([{"faithfulness":.9}])
        def __str__(self): return "scores"
    captured={}
    def modern(**kwargs): captured.update(kwargs); return Scores()
    monkeypatch.setattr(ev,"evaluate",modern); ev.main([])
    cfg=captured["run_config"]
    assert (cfg.max_workers,cfg.timeout,cfg.max_retries)==(1,123,9)
    monkeypatch.setattr(ev,"project_root",tmp_path/"old")
    monkeypatch.setattr(ev,"evaluate",lambda dataset, metrics: Scores())
    ev.main([])
