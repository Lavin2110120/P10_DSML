from __future__ import annotations
from types import SimpleNamespace
import importlib.util
import pandas as pd
import pytest
import rag_test_support  # noqa: F401 - injecte les SDK optionnels avant import des sources
import src.evaluation.evaluate_ragas as evaluation

class FrameScores:
    def to_pandas(self): return pd.DataFrame([{"faithfulness": .9}])

@pytest.mark.unit
def test_rag_system_init_query_and_missing_index(monkeypatch):
    class Store:
        chunks = [{"text":"context"}]
        def __init__(self, client=None, **kwargs): self.client = client
        def search(self, question, k): return [{"text":"evidence"}]
    class Client:
        def generate(self, **kwargs):
            assert "evidence" in kwargs["prompt"] and kwargs["max_tokens"] == 500
            return "answer"
    monkeypatch.setattr(evaluation, "VectorStoreManager", Store)
    monkeypatch.setattr(evaluation, "MistralClient", Client)
    system = evaluation.RAGSystem()
    assert system.query("who?") == {"question":"who?", "answer":"answer", "contexts":["evidence"]}
    class EmptyStore:
        chunks=[]
        def __init__(self, client=None, **kwargs): pass
    monkeypatch.setattr(evaluation, "VectorStoreManager", EmptyStore)
    with pytest.raises(ValueError, match="Index vectoriel vide"): evaluation.RAGSystem()

@pytest.mark.unit
def test_rag_query_without_retrieval_context(monkeypatch):
    class Store:
        chunks=[{}]
        def __init__(self, client=None, **kwargs): pass
        def search(self, question, k): return []
    monkeypatch.setattr(evaluation, "VectorStoreManager", Store)
    monkeypatch.setattr(evaluation, "MistralClient", lambda: object())
    with pytest.raises(ValueError, match="Aucun contexte"): evaluation.RAGSystem().query("question")

@pytest.mark.unit
def test_main_validation_no_results_and_ragas_pipeline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(evaluation, "project_root", tmp_path)
    # ragas est une dépendance optionnelle : main() valide sa présence dès l'entrée.
    monkeypatch.setattr(evaluation, "evaluate", lambda **kwargs: FrameScores())
    questions_path = tmp_path / "data" / "evaluation" / "questions_metier.csv"
    questions_path.parent.mkdir(parents=True)

    class System:
        def query(self, q): return {"answer":"a", "contexts":["c"]}
    class FailingSystem:
        def query(self, q): raise RuntimeError("no answer")

    # Sans CSV, le code source utilise les questions par défaut plutôt que de lever
    # FileNotFoundError ; comme toutes échouent, main() signale l'absence de réponses.
    monkeypatch.setattr(evaluation, "RAGSystem", FailingSystem)
    evaluation.questions_df = None
    with pytest.raises(ValueError, match="Aucune réponse générée"):
        evaluation.main([])

    questions_path.write_text("wrong,col\nx,y\n", encoding="utf8")
    evaluation.questions_df = None
    with pytest.raises(ValueError, match="Colonne.*question.*manquante"):
        evaluation.main([])

    questions_path.write_text("question,ground_truth_expected\n", encoding="utf8")
    evaluation.questions_df = None
    with pytest.raises(ValueError, match="aucune ligne"):
        evaluation.main([])

    questions_path.write_text("question,ground_truth_expected\nq,truth\n", encoding="utf8")
    evaluation.questions_df = None
    with pytest.raises(ValueError, match="Aucune réponse générée"):
        evaluation.main([])

    monkeypatch.setattr(evaluation, "RAGSystem", System)
    monkeypatch.setattr(evaluation, "_make_dataset", lambda rows, with_ground_truth: (rows, True))
    monkeypatch.setattr(evaluation, "_ragas_llm_and_embeddings", lambda: {"llm": object(), "embeddings": object()})
    monkeypatch.setattr(evaluation, "_resolve_metrics", lambda with_ground_truth, models: ["metric"])
    monkeypatch.setattr(evaluation, "evaluate", lambda dataset, metrics: FrameScores())
    result = evaluation.main([])
    assert (tmp_path / "data" / "evaluation" / "ragas_scores.csv").read_text(encoding="utf-8-sig").startswith("faithfulness")
    assert "RÉSULTATS RAGAS" in capsys.readouterr().out
