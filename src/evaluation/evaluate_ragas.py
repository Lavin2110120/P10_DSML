"""Évaluation Ragas du système RAG (Gemini + FAISS).

Charge un jeu de questions métier, interroge le vrai pipeline RAG
(VectorStoreManager + GeminiClient) puis évalue les réponses avec Ragas.
Les scores sont sauvegardés en CSV et JSON dans DATA_DIR.

Usage :
    make evaluate
    # ou
    python -m src.evaluation.evaluate_ragas
"""
from __future__ import annotations

import pandas as pd
import json
import logging
from pathlib import Path

from ragas import EvaluationDataset, evaluate
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

from src.config import (
    DATA_DIR,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    SEARCH_K,
)
from src.llm.gemini_client import GeminiClient
from src.rag.vector_store import VectorStoreManager
from src.ssl_bypass import disable_ssl_verification

# Le bypass TLS doit être appliqué avant tout appel réseau (Google, etc.)
disable_ssl_verification()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_FILE = DATA_DIR / "evaluation" / "questions_metier.csv"
SCORES_CSV = DATA_DIR / "evaluation" / "ragas_scores.csv"
SCORES_JSON = DATA_DIR / "evaluation" / "ragas_scores.json"

METRICS = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]


def load_qa_dataset(path: Path = QUESTIONS_FILE) -> list[dict]:
    """Charge la liste des questions métier depuis le CSV (colonnes question, ground_truth_expected)."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Fichier de questions introuvable : {path}. "
            "Crée un CSV avec au minimum les colonnes 'question' et 'ground_truth_expected'."
        )
    df = pd.read_csv(path)
    if "question" not in df.columns:
        raise ValueError(f"Colonne 'question' manquante dans {path}")
    # ground_truth_expected est le nom utilisé dans ton ancien script ; ground_truth aussi accepté
    gt_col = "ground_truth_expected" if "ground_truth_expected" in df.columns else (
        "ground_truth" if "ground_truth" in df.columns else None
    )
    data = [
        {"question": str(row["question"]), "ground_truth": str(row[gt_col]) if gt_col and pd.notna(row.get(gt_col)) else ""}
        for _, row in df.iterrows()
    ]
    logger.info("%d questions chargées depuis %s", len(data), path)
    return data


def build_ragas_dataset(qa: list[dict]) -> EvaluationDataset:
    """Interroge le vrai pipeline RAG (FAISS + Gemini) pour chaque question."""
    store = VectorStoreManager()
    store.load()  # charge l'index FAISS existant (voir indexer.py)
    llm = GeminiClient()

    samples = []
    for item in qa:
        question = item["question"]
        try:
            # search() renvoie des dicts : on récupère le texte via "text"
            retrieved = store.search(question, top_k=SEARCH_K)
            contexts = [doc.get("text", "") for doc in retrieved if doc.get("text")]
            prompt = (
                "Contexte :\n" + "\n\n".join(contexts) + f"\n\nQuestion : {question}"
            )
            answer = llm.generate_content(prompt)
        except Exception as exc:  # une question en échec ne bloque pas les autres
            logger.warning("Question ignorée (%s) : %s", question[:60], exc)
            continue
        samples.append(
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts, 
                "reference": item.get("ground_truth", ""),
            }
        )
    logger.info("%d échantillons évaluables construits", len(samples))
    return EvaluationDataset.from_list(samples)


def evaluate_with_ragas(dataset: EvaluationDataset):
    """Lance Ragas avec le LLM et les embeddings Google du projet."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, google_api_key=GOOGLE_API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY
    )
    return evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=embeddings)


def main() -> None:
    qa = load_qa_dataset()
    dataset = build_ragas_dataset(qa)
    if not dataset.samples:
        raise RuntimeError(
            "Aucun échantillon évaluable : lance d'abord 'python src/ingestion/indexer.py' "
            "pour créer l'index FAISS, et vérifie GOOGLE_API_KEY dans .env."
        )
    result = evaluate_with_ragas(dataset)
    df = result.to_pandas()
    SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SCORES_CSV, index=False)
    SCORES_JSON.write_text(
        json.dumps(result.scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Scores sauvegardés : %s et %s", SCORES_CSV, SCORES_JSON)
    print(df)


if __name__ == "__main__":
    main()