"""Tests unitaires pour evaluate_ragas.py (sans appels réseau réels).

Le conftest fournit déjà un module 'ragas' minimal ; on l'enrichit ici
avant d'importer le module sous test pour éviter les ImportError.
"""
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Stub de ragas AVANT l'import de src.evaluation.evaluate_ragas
# ---------------------------------------------------------------------------
_ragas = sys.modules.get("ragas")
if _ragas is None:
    _ragas = types.ModuleType("ragas")
    _ragas.__path__ = []
    sys.modules["ragas"] = _ragas


class _EvaluationDataset:
    """Version minimale compatible avec evaluate_ragas.py."""

    def __init__(self, samples):
        self.samples = samples

    @classmethod
    def from_list(cls, rows):
        # Transforme les dicts en objets simples avec attributs
        samples = [types.SimpleNamespace(**row) for row in rows]
        return cls(samples)


_ragas.EvaluationDataset = _EvaluationDataset
if getattr(_ragas, "evaluate", None) is None:
    _ragas.evaluate = MagicMock()

# Stub de ragas.metrics (importé au niveau module dans evaluate_ragas.py)
_metrics = types.ModuleType("ragas.metrics")
for _name in ("Faithfulness", "AnswerRelevancy", "ContextPrecision", "ContextRecall"):
    setattr(_metrics, _name, type(_name, (), {"__init__": lambda self: None}))
sys.modules["ragas.metrics"] = _metrics
_ragas.metrics = _metrics

# Maintenant on peut importer le module sous test en toute sécurité
import src.evaluation.evaluate_ragas as ev


# ---------------------------------------------------------------------------
# load_qa_dataset
# ---------------------------------------------------------------------------

def test_load_qa_dataset_ok(tmp_path):
    csv = tmp_path / "questions.csv"
    csv.write_text(
        "question,ground_truth_expected\n"
        "Quelle est la capitale ?,Paris\n"
        "2+2 ?,4\n",
        encoding="utf-8",
    )
    data = ev.load_qa_dataset(csv)
    assert len(data) == 2
    assert data[0] == {"question": "Quelle est la capitale ?", "ground_truth": "Paris"}
    assert data[1]["ground_truth"] == "4"


def test_load_qa_dataset_ground_truth_alias(tmp_path):
    csv = tmp_path / "q.csv"
    csv.write_text("question,ground_truth\nQ?,A\n", encoding="utf-8")
    data = ev.load_qa_dataset(csv)
    assert data[0]["ground_truth"] == "A"


def test_load_qa_dataset_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        ev.load_qa_dataset(tmp_path / "nope.csv")


def test_load_qa_dataset_missing_question_column(tmp_path):
    csv = tmp_path / "bad.csv"
    csv.write_text("autre,colonne\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Colonne 'question' manquante"):
        ev.load_qa_dataset(csv)


def test_load_qa_dataset_empty_ground_truth(tmp_path):
    csv = tmp_path / "empty.csv"
    csv.write_text("question,ground_truth_expected\nQ?,\n", encoding="utf-8")
    data = ev.load_qa_dataset(csv)
    assert data[0]["ground_truth"] == ""


# ---------------------------------------------------------------------------
# build_ragas_dataset
# ---------------------------------------------------------------------------

def _fake_qa():
    return [
        {"question": "Q1", "ground_truth": "GT1"},
        {"question": "Q2", "ground_truth": "GT2"},
    ]


def test_build_ragas_dataset_success(monkeypatch):
    store = MagicMock()
    store.search.return_value = [{"text": "ctx1"}, {"text": "ctx2"}]
    llm = MagicMock()
    llm.generate_content.return_value = "réponse"

    monkeypatch.setattr(ev, "VectorStoreManager", lambda: store)
    monkeypatch.setattr(ev, "GeminiClient", lambda: llm)

    ds = ev.build_ragas_dataset(_fake_qa())
    assert len(ds.samples) == 2
    s = ds.samples[0]
    assert s.user_input == "Q1"
    assert s.response == "réponse"
    assert s.retrieved_contexts == ["ctx1", "ctx2"]
    assert s.reference == "GT1"


def test_build_ragas_dataset_skips_errors(monkeypatch):
    store = MagicMock()
    store.search.return_value = [{"text": "ctx"}]
    llm = MagicMock()
    # 1ère question échoue (ex: 429), 2ème réussit
    llm.generate_content.side_effect = [RuntimeError("429 quota"), "ok"]

    monkeypatch.setattr(ev, "VectorStoreManager", lambda: store)
    monkeypatch.setattr(ev, "GeminiClient", lambda: llm)

    ds = ev.build_ragas_dataset(_fake_qa())
    assert len(ds.samples) == 1
    assert ds.samples[0].user_input == "Q2"
    assert ds.samples[0].response == "ok"


def test_build_ragas_dataset_empty_contexts(monkeypatch):
    store = MagicMock()
    store.search.return_value = [{"text": ""}, {}]  # aucun texte exploitable
    llm = MagicMock()
    llm.generate_content.return_value = "resp"

    monkeypatch.setattr(ev, "VectorStoreManager", lambda: store)
    monkeypatch.setattr(ev, "GeminiClient", lambda: llm)

    ds = ev.build_ragas_dataset([{"question": "Q", "ground_truth": "G"}])
    assert ds.samples[0].retrieved_contexts == []


# ---------------------------------------------------------------------------
# evaluate_with_ragas
# ---------------------------------------------------------------------------

def test_evaluate_with_ragas_calls_ragas(monkeypatch):
    dataset = MagicMock()
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame({"faithfulness": [0.9]})

    # On remplace la fonction evaluate importée dans le module
    monkeypatch.setattr(ev, "evaluate", lambda **kw: fake_result)

    # Stub des imports internes à la fonction (langchain_google_genai)
    fake_langchain = types.ModuleType("langchain_google_genai")
    fake_langchain.ChatGoogleGenerativeAI = MagicMock()
    fake_langchain.GoogleGenerativeAIEmbeddings = MagicMock()

    with patch.dict(sys.modules, {"langchain_google_genai": fake_langchain}):
        result = ev.evaluate_with_ragas(dataset)

    assert result.to_pandas().iloc[0]["faithfulness"] == 0.9


# ---------------------------------------------------------------------------
# main (intégration avec mocks)
# ---------------------------------------------------------------------------

def test_main_writes_scores(tmp_path, monkeypatch):
    qdir = tmp_path / "evaluation"
    qdir.mkdir()
    questions_csv = qdir / "questions_metier.csv"
    questions_csv.write_text("question,ground_truth_expected\nQ?,A\n", encoding="utf-8")

    scores_csv = qdir / "ragas_scores.csv"
    scores_json = qdir / "ragas_scores.json"

    monkeypatch.setattr(ev, "QUESTIONS_FILE", questions_csv)
    monkeypatch.setattr(ev, "SCORES_CSV", scores_csv)
    monkeypatch.setattr(ev, "SCORES_JSON", scores_json)

    # Mock du pipeline RAG
    store = MagicMock()
    store.search.return_value = [{"text": "ctx"}]
    llm = MagicMock()
    llm.generate_content.return_value = "answer"
    monkeypatch.setattr(ev, "VectorStoreManager", lambda: store)
    monkeypatch.setattr(ev, "GeminiClient", lambda: llm)

    # Mock de l'évaluation Ragas
    fake_result = MagicMock()
    fake_result.to_pandas.return_value = pd.DataFrame({"faithfulness": [0.8]})
    fake_result.scores = {"faithfulness": 0.8}
    monkeypatch.setattr(ev, "evaluate_with_ragas", lambda ds: fake_result)

    ev.main()

    assert scores_csv.exists()
    assert scores_json.exists()
    data = json.loads(scores_json.read_text(encoding="utf-8"))
    assert data["faithfulness"] == 0.8


def test_main_raises_if_no_samples(tmp_path, monkeypatch):
    qdir = tmp_path / "evaluation"
    qdir.mkdir()
    questions_csv = qdir / "questions_metier.csv"
    questions_csv.write_text("question,ground_truth_expected\nQ?,A\n", encoding="utf-8")
    monkeypatch.setattr(ev, "QUESTIONS_FILE", questions_csv)

    # Dataset vide (toutes les questions ont échoué)
    empty_ds = _EvaluationDataset([])
    monkeypatch.setattr(ev, "build_ragas_dataset", lambda qa: empty_ds)

    with pytest.raises(RuntimeError, match="Aucun échantillon"):
        ev.main()