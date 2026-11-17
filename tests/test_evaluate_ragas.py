"""Offline unit tests for src.evaluation.evaluate_ragas; all network SDKs are fake."""
import importlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

# RAGAS is optional at import time. Force a tiny placeholder so an installed
# RAGAS release cannot change import behavior or make test collection network-aware.
_ragas = types.ModuleType("ragas")
_ragas.evaluate = lambda **kwargs: kwargs
sys.modules["ragas"] = _ragas

# The production adapter imports google.genai at module import time. If that
# optional SDK is absent entirely, provide only the import namespace needed for
# loading the module; RAGSystem's constructor is replaced by the offline fixture.
try:
    import google.genai  # noqa: F401
except ImportError:
    _google = sys.modules.get("google")
    if _google is None:
        _google = types.ModuleType("google")
        _google.__path__ = []
        sys.modules["google"] = _google
    _genai = types.ModuleType("google.genai")  # intentionally no Client
    _google.genai = _genai
    sys.modules["google.genai"] = _genai

import src.evaluation.evaluate_ragas as ev


@pytest.fixture(autouse=True)
def reset_globals(monkeypatch, tmp_path):
    ev.project_root = tmp_path
    ev.questions_df = None
    ev.results = []
    ev.dataset = None
    ev.scores = None
    ev.MIN_INTERVAL_S = 0
    ev.GENERATION_MAX_RETRIES = 2
    ev.GENERATION_RETRY_BASE_DELAY = 0
    ev.GENERATION_RETRY_MAX_DELAY = 0
    ev._last_call_time = None
    monkeypatch.setattr(ev.time, "sleep", lambda *_: None)
    monkeypatch.setattr(ev.random, "uniform", lambda *_: 0)

    # RAGSystem's constructor imports production adapters; keep every test offline
    # even when real SDK modules are installed, missing, or only partially present.
    class OfflineGeminiClient:
        def generate_content(self, prompt):
            return "offline answer"

    class OfflineVectorStoreManager:
        def __init__(self):
            self.index = SimpleNamespace(d=ev.config.EMBEDDING_DIM)
        def load(self):
            return None
        def search(self, *args, **kwargs):
            return []

    monkeypatch.setattr(ev, "GeminiClient", OfflineGeminiClient)
    monkeypatch.setattr(ev, "VectorStoreManager", OfflineVectorStoreManager)
    # SDK-wrapper unit tests use an explicit, non-secret placeholder key.
    monkeypatch.setattr(ev.config, "GOOGLE_API_KEY", "unit-test-api-key", raising=False)
    monkeypatch.setattr(ev.config, "GEMINI_API_KEY", None, raising=False)


@pytest.fixture(autouse=True)
def restore_optional_modules():
    prefixes = ("ragas", "datasets", "langchain_core", "langchain_google_genai")
    before = {name: mod for name, mod in sys.modules.items() if name == prefixes[0] or any(name.startswith(p + ".") for p in prefixes) or name in prefixes[1:]}
    yield
    for name in list(sys.modules):
        if name == "ragas" or any(name.startswith(p + ".") for p in prefixes) or name in prefixes[1:]:
            sys.modules.pop(name, None)
    sys.modules.update(before)


def module(name, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    return mod


def test_throttle_and_error_classification(monkeypatch):
    ticks = iter([10.0, 10.25, 10.75])
    monkeypatch.setattr(ev.time, "monotonic", lambda: next(ticks))
    sleeps = []
    monkeypatch.setattr(ev.time, "sleep", sleeps.append)
    ev._last_call_time = None
    ev._throttle(0)
    ev._throttle(.5)
    assert sleeps == [.25]
    assert ev._is_retryable_error(Exception("503 RESOURCE_EXHAUSTED"))
    assert ev._is_retryable_error(SimpleNamespace(status_code=429))
    assert ev._is_retryable_error(Exception("high demand"))
    assert not ev._is_retryable_error(Exception("bad input"))
    assert ev._is_rate_limit_error is ev._is_retryable_error


@pytest.mark.parametrize("exc, expected", [
    (Exception("oops"), None),
    (SimpleNamespace(status_code=429), "429"),
    (SimpleNamespace(status=None, response=SimpleNamespace(status_code=503)), "503"),
    (SimpleNamespace(code=lambda: 503), "503"),
    (SimpleNamespace(code=lambda: (_ for _ in ()).throw(RuntimeError())), None),
])
def test_http_status(exc, expected):
    assert ev._http_status(exc) == expected


def test_retry_after_header_seconds_and_date(monkeypatch):
    assert ev._retry_after_seconds(Exception()) is None
    assert ev._retry_after_seconds(SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "-2"}))) == 0
    assert ev._retry_after_seconds(SimpleNamespace(response=SimpleNamespace(headers={"retry-after": "2.5"}))) == 2.5
    monkeypatch.setattr(ev, "parsedate_to_datetime", lambda _: ev.datetime.now(ev.timezone.utc).replace(year=2026))
    # Date parsing succeeds for a parseable HTTP date; negative waits clamp to zero.
    exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "Sun, 06 Nov 1994 08:49:37 GMT"}))
    assert ev._retry_after_seconds(exc) == 0
    monkeypatch.setattr(ev, "parsedate_to_datetime", lambda _: (_ for _ in ()).throw(ValueError()))
    bad = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "nonsense"}))
    assert ev._retry_after_seconds(bad) is None


def test_generate_retry_interfaces(monkeypatch):
    class Client:
        def __init__(self): self.calls = 0
        def generate_content(self, prompt):
            self.calls += 1
            if self.calls == 1: raise Exception("503 unavailable")
            return " answer "
    client = Client()
    assert ev._generate_with_retry(client, "q") == "answer"
    assert client.calls == 2

    class Positional:
        def generate_content(self, prompt, /): return " positional "
    assert ev._generate_with_retry(Positional(), "q") == "positional"

    class Old:
        def generate(self, prompt): return None
    assert ev._generate_with_retry(Old(), "q") == ""
    class Broken:
        def generate_content(self, prompt): raise ValueError("invalid")
    with pytest.raises(ValueError): ev._generate_with_retry(Broken(), "q")


def test_generate_retry_exhaustion_and_signature_typeerror(monkeypatch):
    ev.GENERATION_MAX_RETRIES = 1
    calls = []
    class Retry:
        def generate_content(self, prompt):
            calls.append(1)
            raise Exception("429 too many requests")
    with pytest.raises(Exception): ev._generate_with_retry(Retry(), "q")
    assert len(calls) == 2
    class TypeErrorClient:
        def generate_content(self, prompt): raise TypeError("unrelated internal issue")
    with pytest.raises(TypeError, match="unrelated"): ev._generate_with_retry(TypeErrorClient(), "q")


def test_rag_system_context_adaptation(monkeypatch):
    class Client:
        def generate_content(self, prompt):
            assert "one" in prompt and "two" in prompt
            return " answer "
    class Store:
        def __init__(self): self.index = SimpleNamespace(d=ev.config.EMBEDDING_DIM)
        def load(self): pass
        def search(self, question, top_k):
            assert top_k == 2
            return [{"text": " one ", "content": "ignored"}, {"text": " two "},
                    SimpleNamespace(page_content=" three "), {"page_content": ""}, None]
    monkeypatch.setattr(ev, "GeminiClient", Client)
    monkeypatch.setattr(ev, "VectorStoreManager", Store)
    monkeypatch.setattr(ev, "generate_structured_answer", lambda question, context, chunks, client: {
        "answer": "answer", "sources": [{"text": "one"}], "grounded": True,
    })
    sys = ev.RAGSystem(top_k=2, max_chunk_chars=3)
    assert sys.query("q") == {"question": "q", "answer": "answer", "contexts": ["one", "two", "thr"], "mode": "structured"}


@pytest.mark.parametrize("retrieved", [None, [], [{"text": "  "}], [object()]])
def test_rag_system_empty_context(retrieved, monkeypatch):
    class Store:
        def __init__(self): self.index = SimpleNamespace(d=ev.config.EMBEDDING_DIM)
        def load(self): pass
        def search(self, *args, **kwargs): return retrieved
    monkeypatch.setattr(ev, "VectorStoreManager", Store)
    with pytest.raises(ValueError, match="Aucun chunk non vide trouvé"): ev.RAGSystem().query("q")


def test_validate_questions_df():
    valid = pd.DataFrame([{"question": "q", "ground_truth_expected": "a", "category": "simple"}])
    assert ev._validate_questions_df(valid) is valid
    cases = [
        (pd.DataFrame({"question": ["q"]}), "Colonnes manquantes"),
        (valid.iloc[:0], "vide"),
        (valid.assign(category="unknown"), "Catégories inconnues"),
        (valid.assign(question=None), "ne doivent pas être vides"),
    ]
    for frame, message in cases:
        with pytest.raises(ValueError, match=message): ev._validate_questions_df(frame)


def test_load_questions_df_csv_default_and_cached(monkeypatch, tmp_path):
    ev.questions_df = pd.DataFrame({"question": ["cached"], "ground_truth_expected": ["truth"], "category": ["simple"]})
    assert ev._load_questions_df().iloc[0, 0] == "cached"
    ev.questions_df = None
    path = tmp_path / "data/evaluation/questions_metier.csv"
    path.parent.mkdir(parents=True)
    path.write_text("question,ground_truth_expected,category\nQ,A,simple\n")
    assert ev._load_questions_df().iloc[0, 0] == "Q"
    ev.questions_df = None
    path.write_text("wrong\nx\n")
    with pytest.raises(ValueError, match="Colonnes manquantes"): ev._load_questions_df()
    ev.questions_df = None
    path.write_text("question,ground_truth_expected,category\n")
    with pytest.raises(ValueError, match="CSV des questions est vide"): ev._load_questions_df()
    path.unlink()
    ev.questions_df = None
    with pytest.raises(FileNotFoundError, match="Fichier de questions métier absent"):
        ev._load_questions_df()


def test_dataset_modern_and_legacy(monkeypatch):
    rows = [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"], "reference": "gt"},
            {"user_input": "q2", "response": "a2", "retrieved_contexts": []}]
    class Sample:
        def __init__(self, **kwargs): self.values = kwargs
    class Dataset:
        def __init__(self, samples): self.samples = samples
    schema = module("ragas.dataset_schema", EvaluationDataset=Dataset, SingleTurnSample=Sample)
    assert ev._make_dataset(rows, True)[0].samples[0].values["reference"] == "gt"
    assert "reference" not in ev._make_dataset(rows, False)[0].samples[0].values
    sys.modules.pop("ragas.dataset_schema", None)
    class Legacy:
        @staticmethod
        def from_list(rows): return rows
    module("datasets", Dataset=Legacy)
    old, modern = ev._make_dataset(rows, True)
    assert not modern and old[0]["ground_truth"] == "gt"
    assert "ground_truth" not in ev._make_dataset(rows, False)[0][1]
    sys.modules.pop("datasets", None)
    real_import = __import__
    def no_datasets(name, *args, **kwargs):
        if name == "datasets": raise ImportError("absent")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr("builtins.__import__", no_datasets)
    with pytest.raises(ImportError, match="Installez ragas"):
        ev._make_dataset(rows, False)


def test_dataset_schema_missing_attr_fallback(monkeypatch):
    module("ragas.dataset_schema")
    assert ev._ragas_dataset_api() == (None, None)


def test_call_supported_kwargs():
    def f(a, *, b=0): return a+b
    assert ev._call_with_supported_kwargs(f, {"a": 2, "b": 3, "c": 9}) == 5
    class Kw:
        def __init__(self, **kw): self.kw = kw
    assert ev._call_with_supported_kwargs(Kw, {"x": 1}).kw == {"x": 1}


def test_throttle_callback_present_and_missing(monkeypatch):
    core = module("langchain_core")
    callbacks = module("langchain_core.callbacks", BaseCallbackHandler=type("Base", (), {}))
    cb = ev._make_throttle_callback()
    cb.on_chat_model_start(); cb.on_llm_start()
    sys.modules.pop("langchain_core.callbacks", None)
    with pytest.warns(None) if False else _no_warn_context():
        assert ev._make_throttle_callback() is None


class _no_warn_context:
    def __enter__(self): return self
    def __exit__(self, *args): return False


def _install_gemini_sdk(monkeypatch, chat_cls, embed_cls):
    module("langchain_google_genai", ChatGoogleGenerativeAI=chat_cls,
           GoogleGenerativeAIEmbeddings=embed_cls)
    module("ragas.llms", LangchainLLMWrapper=lambda obj: ("llm", obj))
    module("ragas.embeddings", LangchainEmbeddingsWrapper=lambda obj: ("embed", obj))


def test_gemini_wrappers_rest_and_env_fallback(monkeypatch):
    seen = []
    class Chat:
        def __init__(self, model, temperature, transport, api_key, google_api_key): seen.append(("c", transport, api_key, google_api_key))
    class Embed:
        def __init__(self, model, transport, google_api_key): seen.append(("e", transport, google_api_key))
    _install_gemini_sdk(monkeypatch, Chat, Embed)
    assert ev._ragas_llm_and_embeddings()["llm"][0] == "llm"
    assert [x[1] for x in seen] == ["rest", "rest"]
    monkeypatch.setattr(ev, "_make_throttle_callback", lambda: None)
    ev._ragas_llm_and_embeddings()


def test_gemini_missing_dependencies_and_key(monkeypatch):
    real_import = __import__
    def fake_import(name, *a, **kw):
        if name == "langchain_google_genai": raise ImportError("absent")
        return real_import(name, *a, **kw)
    with monkeypatch.context() as m:
        m.setattr("builtins.__import__", fake_import)
        with pytest.raises(ImportError, match="langchain-google-genai"): ev._ragas_llm_and_embeddings()
    # Complete harmless modules keep this test independent of all installed SDKs.
    _install_gemini_sdk(monkeypatch, lambda **kw: object(), lambda **kw: object())
    # Config releases expose different aliases; clear any in-memory/env key
    # without depending on a particular attribute being present.
    monkeypatch.setattr(ev.config, "GOOGLE_API_KEY", None, raising=False)
    monkeypatch.setattr(ev.config, "GEMINI_API_KEY", None, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="clé Gemini"): ev._ragas_llm_and_embeddings()


def test_gemini_no_transport_signature_and_runtime_fallback(monkeypatch):
    created = []
    class Chat:
        def __init__(self, model, temperature, api_key, google_api_key):
            created.append("chat"); self.callbacks = []
    class Embed:
        def __init__(self, model, google_api_key): created.append("embed")
    _install_gemini_sdk(monkeypatch, Chat, Embed)
    monkeypatch.setattr(ev, "_make_throttle_callback", lambda: object())
    assert ev._ragas_llm_and_embeddings()["embeddings"][0] == "embed"
    assert created == ["chat", "embed"]

    class RuntimeChat:
        def __init__(self, model, temperature, transport=None, api_key=None, google_api_key=None):
            if transport: raise ValueError("transport refused")
    class RuntimeEmbed:
        def __init__(self, model, transport=None, google_api_key=None):
            if transport: raise TypeError("transport refused")
    _install_gemini_sdk(monkeypatch, RuntimeChat, RuntimeEmbed)
    ev._ragas_llm_and_embeddings()


def test_resolve_metrics_aliases_classes_instances_and_errors(monkeypatch):
    models = {"llm": object(), "embeddings": object()}
    class Faith:
        def __init__(self, llm): self.llm = llm
    class Relevancy:
        def __init__(self, llm, embeddings): self.args = (llm, embeddings)
    class Precision:
        def __init__(self, llm): self.llm = llm
    class Recall:
        def __init__(self, llm): self.llm = llm
    class Correct:
        def __init__(self, llm): self.llm = llm
    module("ragas.metrics", Faithfulness=Faith, ResponseRelevancy=Relevancy,
           LLMContextPrecisionWithReference=Precision, LLMContextRecall=Recall, FactualCorrectness=Correct)
    module("ragas.metrics.collections")
    metrics = ev._resolve_metrics(True, models)
    assert len(metrics) == 5 and metrics[0].llm is models["llm"]
    instance = SimpleNamespace()
    module("ragas.metrics", faithfulness=instance, answer_relevancy=object(),
           ContextPrecisionWithoutReference=SimpleNamespace())
    # Legacy prebuilt instances are wired; immutable objects warn but remain metrics.
    assert len(ev._resolve_metrics(False, models)) == 3
    assert instance.llm is models["llm"]
    module("ragas.metrics")
    with pytest.raises(RuntimeError, match="Aucune métrique"): ev._resolve_metrics(False, models)


def test_resolve_metrics_missing_and_unsettable(monkeypatch):
    module("ragas.metrics")
    sys.modules.pop("ragas.metrics.collections", None)
    # neither parent package needs to exist for import_module's failure branch
    with pytest.raises(RuntimeError): ev._resolve_metrics(False, {"llm": object(), "embeddings": object()})


def test_signature_failures_and_callback_assignment(monkeypatch):
    real_signature = ev.inspect.signature
    def bad_signature(obj):
        if obj is opaque: raise ValueError("opaque")
        return real_signature(obj)
    def opaque(): pass
    monkeypatch.setattr(ev.inspect, "signature", bad_signature)
    assert ev._call_with_supported_kwargs(opaque, {"x": 1}) is None
    monkeypatch.setattr(ev.inspect, "signature", real_signature)

    class OpaqueChat:
        @property
        def __signature__(self): raise ValueError("opaque")
        def __call__(self, **kwargs): return SimpleNamespace()
    _install_gemini_sdk(monkeypatch, OpaqueChat(), OpaqueChat())
    monkeypatch.setattr(ev, "_make_throttle_callback", lambda: object())
    ev._ragas_llm_and_embeddings()

    class NoCallbacks:
        def __setattr__(self, name, value):
            if name == "callbacks": raise RuntimeError("read only")
            object.__setattr__(self, name, value)
    class Chat:
        def __init__(self, model, temperature, transport, api_key, google_api_key): return None
    # Return a no-callback-capable instance from factory.
    class ChatFactory:
        def __init__(self, model, temperature, transport, api_key, google_api_key): self.instance = NoCallbacks()
    _install_gemini_sdk(monkeypatch, ChatFactory, lambda model, transport, google_api_key: object())
    ev._ragas_llm_and_embeddings()


def test_remaining_io_and_metrics_error_branches(monkeypatch, tmp_path):
    with pytest.raises(ImportError, match="ragas est requis"):
        with monkeypatch.context() as m:
            real_import = __import__
            def no_metrics(name, *args, **kwargs):
                if name == "ragas.metrics": raise ImportError("missing")
                return real_import(name, *args, **kwargs)
            m.setattr("builtins.__import__", no_metrics)
            ev._resolve_metrics(False, {"llm": 1, "embeddings": 2})

    bad_frame = object()
    assert ev._scores_frame(bad_frame).empty
    ev._write_reports([{"question": "q", "category": "c", "score": 1.0}],
                      pd.DataFrame({"score": [0.5]}), tmp_path / "report")
    bad_json = tmp_path / "not-object.json"
    bad_json.write_text("[]")
    assert ev._load_response_cache(bad_json) == {}

    class BrokenPath:
        def open(self, *a, **k): raise OSError("cannot read")
    assert ev._load_jsonl_response_cache(BrokenPath()) == {}


def test_main_sleep_exception_unlimited_and_legacy_signature(monkeypatch):
    calls = []
    class Sys:
        def query(self, q):
            if q == "bad": raise RuntimeError("query failed")
            return {"answer": "a", "contexts": ["x"]}
    monkeypatch.setattr(ev, "RAGSystem", lambda **kw: Sys())
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, gt))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {"llm": 1, "embeddings": 2})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: [])
    def old_evaluate(dataset, metrics, max_workers=None, timeout=None, max_retries=None, max_wait=None, raise_exceptions=None):
        calls.append((max_workers, timeout, max_retries, max_wait))
        return pd.DataFrame({"score": [1]})
    monkeypatch.setattr(ev, "evaluate", old_evaluate)
    monkeypatch.setattr(ev, "_parse_args", lambda argv: SimpleNamespace(
        min_interval=0, categories=None, limit=None, top_k=3, max_chunk_chars=None,
        sleep=1, max_workers=1, ragas_timeout=12, ragas_retries=2, raise_exceptions=False))
    slept = []
    monkeypatch.setattr(ev.time, "sleep", lambda seconds: slept.append(seconds))
    ev.questions_df = pd.DataFrame({"question": ["bad", "good", "good2"], "ground_truth_expected": ["gt"] * 3, "category": ["simple"] * 3})
    ev.main([])
    assert slept == [1, 1] and len(ev.results) == 2
    # RunConfig is deliberately unavailable in this isolated test; old direct options are not passed.
    assert calls[0] == (None, None, None, None)


def test_import_without_ragas_and_restore(monkeypatch):
    original = __import__
    original_evaluate = ev.evaluate
    with monkeypatch.context() as m:
        def fail_ragas(name, *args, **kwargs):
            if name == "ragas": raise ImportError("optional package absent")
            return original(name, *args, **kwargs)
        m.setattr("builtins.__import__", fail_ragas)
        importlib.reload(ev)
        assert ev.evaluate is None

    # Restoration must not depend on whether RAGAS is installed on this host.
    # Supply a minimal optional-module API, reload, and restore the pre-test
    # binding so the rest of the suite observes its original environment.
    restored_evaluate = lambda **kwargs: kwargs
    ragas_stub = types.ModuleType("ragas")
    ragas_stub.evaluate = restored_evaluate
    monkeypatch.setitem(sys.modules, "ragas", ragas_stub)
    importlib.reload(ev)
    assert ev.evaluate is restored_evaluate
    ev.evaluate = original_evaluate


def test_runtime_transport_failure_after_signature_fallback(monkeypatch):
    class RejectChat:
        def __init__(self, model, temperature, api_key, google_api_key):
            raise ValueError("rejected")
    class Embed:
        def __init__(self, model, google_api_key): pass
    _install_gemini_sdk(monkeypatch, RejectChat, Embed)
    with pytest.raises(ValueError, match="rejected"):
        ev._ragas_llm_and_embeddings()


def test_callback_assignment_exception(monkeypatch):
    class ReadOnlyChat:
        def __init__(self, model, temperature, transport, api_key, google_api_key): pass
        def __setattr__(self, name, value):
            if name == "callbacks": raise RuntimeError("read only")
            object.__setattr__(self, name, value)
    class Embed:
        def __init__(self, model, transport, google_api_key): pass
    _install_gemini_sdk(monkeypatch, ReadOnlyChat, Embed)
    monkeypatch.setattr(ev, "_make_throttle_callback", lambda: object())
    assert ev._ragas_llm_and_embeddings()["llm"][0] == "llm"


def test_main_truncation_to_empty_and_legacy_ragas_api(monkeypatch):
    _main_stubs(monkeypatch)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    monkeypatch.setattr(ev, "_parse_args", lambda argv: SimpleNamespace(
        min_interval=0, categories=None, limit=None, top_k=3, max_chunk_chars=0,
        sleep=0, max_workers=1, ragas_timeout=10, ragas_retries=2, raise_exceptions=True))
    with pytest.raises(ValueError, match="Aucune réponse générée"):
        ev.main([])

    # RAGAS RunConfig exists, but this old evaluate API only accepts direct options.
    module("ragas.run_config", RunConfig=lambda **kw: SimpleNamespace(**kw))
    seen = {}
    def legacy(dataset, metrics, timeout=None, max_retries=None, max_wait=None, raise_exceptions=None):
        seen.update(timeout=timeout, max_retries=max_retries, max_wait=max_wait)
        return pd.DataFrame({"score": [1]})
    _main_stubs(monkeypatch, legacy)
    monkeypatch.setattr(ev, "_parse_args", lambda argv: SimpleNamespace(
        min_interval=0, categories=None, limit=None, top_k=3, max_chunk_chars=20,
        sleep=0, max_workers=1, ragas_timeout=10, ragas_retries=2, raise_exceptions=True))
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    ev.main([])
    assert seen == {"timeout": 10, "max_retries": 2, "max_wait": ev.DEFAULT_RAGAS_MAX_WAIT}


def test_retry_limit_ragas_root_runconfig_fallback(monkeypatch):
    # A failed import of ragas.run_config falls back to the package's RunConfig export,
    # including when the first evaluation hits 429 and requires a retry.
    module("ragas", evaluate=lambda **kwargs: None, RunConfig=lambda **kw: SimpleNamespace(**kw))
    sys.modules.pop("ragas.run_config", None)
    calls = []
    def evaluate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1: raise Exception("429 rate limit")
        return pd.DataFrame({"score": [1]})
    _main_stubs(monkeypatch, evaluate)
    # Exercise package-root RunConfig import on both initial config and retry.
    sys.modules.pop("ragas.run_config", None)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    ev.main([])
    assert len(calls) == 2


def test_script_entrypoint_catches_failure(monkeypatch):
    import runpy
    package = types.ModuleType("ragas")
    package.evaluate = None
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "ragas", package)
        with pytest.raises(SystemExit) as exc:
            runpy.run_path(str(Path(ev.__file__)), run_name="__main__")
        assert exc.value.code == 1


def test_retry_loop_zero_attempts(monkeypatch):
    monkeypatch.setattr(ev, "GENERATION_MAX_RETRIES", -1)
    assert ev._generate_with_retry(object(), "q") == ""


def test_parse_args_and_errors():
    a = ev._parse_args(["--categories", " Simple,", "complexe", "--limit", "0", "--min-interval", "0"])
    assert a.categories == ["Simple", "complexe"] and a.limit == 0
    for argv in (["--limit", "-1"], ["--top-k", "0"], ["--sleep", "-1"], ["--min-interval", "-1"]):
        with pytest.raises(SystemExit): ev._parse_args(argv)


def test_response_cache_read_write_jsonl(tmp_path, monkeypatch):
    missing = tmp_path / "missing.json"
    assert ev._load_response_cache(missing) == {}
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"q": {"answer": "a", "contexts": ["c"]}, "bad": [], "x": {"answer": ""}}))
    assert list(ev._load_response_cache(path)) == ["q"]
    path.write_text("[")
    assert ev._load_response_cache(path) == {}
    ev._save_response_cache(path, {"q": {"answer": "é", "contexts": ["c"]}})
    assert ev._load_response_cache(path)["q"]["answer"] == "é"
    jpath = tmp_path / "answers.jsonl"
    assert ev._load_jsonl_response_cache(jpath) == {}
    jpath.write_text('{"question":"q","response":{"answer":"a","contexts":["c"]}}\nnull\nnope\n{"question":"bad","answer":"","contexts":[]}\n')
    assert ev._load_jsonl_response_cache(jpath)["q"]["answer"] == "a"
    ev._append_jsonl_response(jpath, "q2", {"answer": "b", "contexts": ["d"]})
    assert "q2" in ev._load_jsonl_response_cache(jpath)


def test_score_frame_and_reports(tmp_path):
    class Scores:
        def to_pandas(self): return pd.DataFrame({"faithfulness": [.5, .75]})
    assert list(ev._scores_frame(Scores()).columns) == ["faithfulness"]
    frame = pd.DataFrame({"x": [1]})
    assert ev._scores_frame(frame).equals(frame)
    assert ev._scores_frame({"x": [2]}).iloc[0, 0] == 2
    ev._write_reports([
        {"question": "q|", "category": "a", "answer": "ans", "contexts": "[]"},
        {"question": "q2", "category": "a", "answer": "ans", "contexts": "[]"},
    ], pd.DataFrame({"score": [.5]}), tmp_path)
    assert (tmp_path / "ragas_detail_par_question.csv").exists()
    assert "question_count" in pd.read_csv(tmp_path / "ragas_scores_par_categorie.csv").columns
    assert "a" in (tmp_path / "ragas_scores_par_categorie.md").read_text()
    ev._write_reports([], [], tmp_path / "empty")


def _main_stubs(monkeypatch, evaluate_func=None):
    monkeypatch.setattr(ev, "RAGSystem", lambda **kw: SimpleNamespace(query=lambda q: {"answer": " A ", "contexts": [" c "]}))
    monkeypatch.setattr(ev, "_make_dataset", lambda rows, gt: (rows, gt))
    monkeypatch.setattr(ev, "_ragas_llm_and_embeddings", lambda: {"llm": 1, "embeddings": 2})
    monkeypatch.setattr(ev, "_resolve_metrics", lambda gt, models: ["metric"])
    monkeypatch.setattr(ev, "evaluate", evaluate_func or (lambda **kwargs: pd.DataFrame({"score": [.8]})))
    module("ragas.run_config", RunConfig=lambda **kw: SimpleNamespace(**kw))


def test_main_happy_path_filtered_legacy_category_and_reports(monkeypatch, capsys):
    _main_stubs(monkeypatch)
    ev.questions_df = pd.DataFrame({"question": [" q ", "drop", "nan", ""], "categorie": ["simple", "complexe", "simple", "simple"], "ground_truth_expected": ["truth"] * 4})
    result = ev.main(["--categories", "simple", "--limit", "1", "--max-workers", "2"])
    assert isinstance(result, pd.DataFrame)
    assert ev.results[0]["reference"] == "truth"
    assert (ev.project_root / "reports/ragas_input.csv").exists()
    assert "RÉSULTATS RAGAS" in capsys.readouterr().out


def test_main_no_ragas_wrong_columns_no_responses(monkeypatch):
    monkeypatch.setattr(ev, "evaluate", None)
    with pytest.raises(ImportError): ev.main([])
    monkeypatch.setattr(ev, "evaluate", lambda **k: None)
    ev.questions_df = pd.DataFrame({"wrong": ["x"]})
    with pytest.raises(ValueError, match="Colonnes manquantes"): ev.main([])
    ev.questions_df = pd.DataFrame({"question": [" ", float("nan")], "ground_truth_expected": ["gt", "gt"], "category": ["simple", "simple"]})
    with pytest.raises(ValueError, match="ne doivent pas être vides"): ev.main([])


def test_main_cached_bad_rows_partial_gt_and_empty_after_truncate(monkeypatch):
    _main_stubs(monkeypatch)
    ev.questions_df = pd.DataFrame({"question": ["cached", "bad", "truncate", "ok"], "category": ["simple"] * 4, "ground_truth_expected": ["gt", "", "", "x"]})
    monkeypatch.setattr(ev, "_load_response_cache", lambda p: {"cached": {"answer": " ans ", "contexts": [" c "]}})
    monkeypatch.setattr(ev, "_load_jsonl_response_cache", lambda p: {})
    class Sys:
        def query(self, q):
            if q == "bad": return {"answer": "", "contexts": ["c"]}
            if q == "truncate": return {"answer": "a", "contexts": [" "]}
            return {"answer": " a ", "contexts": " text "}
    monkeypatch.setattr(ev, "RAGSystem", lambda **kw: Sys())
    ev.main(["--max-chunk-chars", "1"])
    assert [r["user_input"] for r in ev.results] == ["cached", "ok"]
    assert all("reference" not in r for r in ev.results)


def test_main_evaluate_legacy_options_and_retry_rate_limit(monkeypatch):
    attempts = []
    def evaluate(dataset, metrics, **kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1: raise Exception("429 rate limit")
        return pd.DataFrame({"score": [.7]})
    _main_stubs(monkeypatch, evaluate)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    ev.main([])
    assert len(attempts) == 2 and "run_config" in attempts[0]


def test_main_run_config_missing_and_nonretry_error(monkeypatch):
    _main_stubs(monkeypatch)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    sys.modules.pop("ragas.run_config", None)
    # root ragas has no RunConfig; both imports fail, and main logs fallback.
    ev.main([])
    def fail(**kwargs): raise RuntimeError("fatal")
    _main_stubs(monkeypatch, fail)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    with pytest.raises(RuntimeError, match="fatal"): ev.main([])


def test_main_runconfig_eval_signature_inspection_failure(monkeypatch):
    _main_stubs(monkeypatch)
    ev.questions_df = pd.DataFrame({"question": ["q"], "ground_truth_expected": ["gt"], "category": ["simple"]})
    # Signature lookup failure branch on a callable object.
    class Opaque:
        @property
        def __signature__(self): raise ValueError("opaque")
        def __call__(self, **kwargs): return pd.DataFrame({"score": [.2]})
    monkeypatch.setattr(ev, "evaluate", Opaque())
    ev.main([])



def test_category_report_and_metric_cross_tab(tmp_path):
    """Category aggregation and cross-tab are produced by _write_reports."""
    rows = [
        {"question": "Q1", "category": "simple", "answer": "A", "contexts": "[]"},
        {"question": "Q2", "category": "simple", "answer": "B", "contexts": "[]"},
        {"question": "Q3", "category": "complexe", "answer": "C", "contexts": "[]"},
    ]
    scores = pd.DataFrame({"faithfulness": [0.2, 0.6, 0.8], "answer_relevancy": [0.4, 0.8, 1.0]})
    ev._write_reports(rows, scores, tmp_path)
    grouped = pd.read_csv(tmp_path / "ragas_scores_par_categorie.csv").set_index("category")
    assert grouped.loc["simple", "question_count"] == 2
    assert grouped.loc["simple", "faithfulness"] == pytest.approx(0.4)
    assert grouped.loc["complexe", "answer_relevancy"] == pytest.approx(1.0)
    cross = pd.read_csv(tmp_path / "ragas_scores_croises_categorie_metrique.csv", index_col=0)
    assert cross.loc["simple", "faithfulness"] == pytest.approx(0.4)
    assert cross.loc["complexe", "answer_relevancy"] == pytest.approx(1.0)
