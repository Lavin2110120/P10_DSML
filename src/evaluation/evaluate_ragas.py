"""Évaluation RAGAS avec génération et recherche Mistral.

Le fichier CSV data/evaluation/questions_metier.csv est facultatif. Sans CSV,
les questions par défaut sont évaluées sans référence (ground truth).
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import re
import random
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from rag.ssl_bypass import disable_ssl_verification
disable_ssl_verification()

import pandas as pd

try:
    from ragas import evaluate
except ImportError:  # permet l'import du module sans la dépendance optionnelle
    evaluate = None

# Makefile runs this module with PYTHONPATH=src, so use the source-root
# package names (rag/llm) rather than the repository-root ``src.*`` names.
from rag import config
from rag.vector_store import VectorStoreManager
from llm.mistral_client import MistralClient

logger = logging.getLogger(__name__)
project_root = Path(getattr(config, "PROJECT_ROOT", Path(__file__).resolve().parents[2]))

DEFAULT_QUESTIONS = [
    "Quels sont les principaux sujets traités dans les documents ?",
    "Résumez le contenu principal des documents disponibles.",
    "Quelles informations clés peut-on retenir de ces documents ?",
]

# Globales conservées pour compatibilité avec les tests qui les remplacent.
questions_df: pd.DataFrame | None = None
results: list[dict[str, Any]] = []
dataset = None
scores = None

# Appels Mistral plafonnés globalement : l'horloge monotone évite les dérives.
MIN_INTERVAL_S = max(0.0, float(os.getenv("MIN_INTERVAL_S", "12")))
BACKOFF_S = [30, 60, 120, 180, 300, 300]
RAGAS_MAX_RETRIES = max(0, int(os.getenv("RAGAS_MAX_RETRIES", str(len(BACKOFF_S)))))
RAGAS_RETRY_BASE_DELAY = 30.0  # conservé pour compatibilité avec d'anciens imports
RAGAS_RETRY_MAX_DELAY = 300.0
RAGAS_QUESTION_DELAY = 0.0
DEFAULT_MAX_WORKERS = 1
DEFAULT_RAGAS_TIMEOUT = 300
DEFAULT_RAGAS_RETRIES = 10
DEFAULT_RAGAS_MAX_WAIT = 120
DEFAULT_TOP_K = 3
DEFAULT_MAX_CHUNK_CHARS = 800
_throttle_lock = threading.Lock()
_last_call_time: float | None = None


def _throttle(interval: float | None = None) -> None:
    """Attend jusqu'à l'intervalle minimal avant un appel chat/completions."""
    global _last_call_time
    minimum = MIN_INTERVAL_S if interval is None else max(0.0, float(interval))
    with _throttle_lock:
        now = time.monotonic()
        wait = 0.0 if _last_call_time is None else minimum - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_call_time = now


def _is_retryable_error(exc: Exception) -> bool:
    """Repère statuts HTTP transitoires et codes Mistral dans erreurs enveloppées."""
    message = str(exc).lower()
    response = getattr(exc, "response", None)
    status = (getattr(exc, "status_code", None) or getattr(exc, "status", None)
              or getattr(response, "status_code", None))
    if str(status) in {"429", "500", "502", "503", "504"}:
        return True
    if re.search(r"\bstatus\s*(?:code\s*)?(?:429|500|502|503|504)\b", message):
        return True
    if re.search(r"\b(?:1300|3505)\b", message):
        return True
    return any(token in message for token in (
        "rate_limited", "rate-limited", "rate limit", "too many requests",
        "out_of_capacity", "out of capacity", "over capacity", "capacity exceeded",
        "service unavailable", "temporarily unavailable",
    ))


def _is_rate_limit_error(exc: Exception) -> bool:
    """Alias historique, désormais vrai pour toutes les erreurs réessayables."""
    return _is_retryable_error(exc)


def _generate_with_retry(client, prompt: str):
    attempts = RAGAS_MAX_RETRIES + 1
    for attempt in range(attempts):
        logger.info("Appel de génération Mistral, tentative %d/%d.", attempt + 1, attempts)
        _throttle()
        try:
            generate = getattr(client, "generate", None)
            if generate is None:
                generate = client.generate_response
            return generate(prompt=prompt, max_tokens=500)
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt >= RAGAS_MAX_RETRIES:
                if _is_retryable_error(exc):
                    logger.warning("Échec après %d tentatives pour l'appel Mistral : %s", attempt + 1, exc)
                raise
            delay_base = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            delay = delay_base + random.uniform(0, 10)
            logger.warning("Échec transitoire Mistral (tentative %d/%d) ; nouvel essai dans %.1f s : %s",
                           attempt + 1, attempts, delay, exc)
            time.sleep(delay)


class RAGSystem:
    def __init__(self, top_k: int = DEFAULT_TOP_K, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS):
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        # search() dépend du client d'embeddings : il faut l'injecter au magasin.
        self.client = MistralClient()
        try:
            store_params = inspect.signature(VectorStoreManager).parameters.values()
            accepts_client = any(p.name == "client" or p.kind == inspect.Parameter.VAR_KEYWORD
                                 for p in store_params)
        except (TypeError, ValueError):
            accepts_client = True
        self.vector_store = (VectorStoreManager(client=self.client) if accepts_client
                             else VectorStoreManager())
        chunks = getattr(self.vector_store, "chunks",
                         getattr(self.vector_store, "document_chunks", None))
        if chunks is not None and len(chunks) == 0:
            raise ValueError(
                "Index vectoriel vide : construisez l'index d'abord "
                "(make index / make vectorstore)"
            )

    def query(self, question: str) -> dict[str, Any]:
        retrieved = self.vector_store.search(question, k=self.top_k)
        logger.info("search(%r) -> %d résultat(s)", question[:50], len(retrieved))
        contexts: list[str] = []
        for item in retrieved:
            if isinstance(item, dict):
                metadata = item.get("metadata") or {}
                text = item.get("text") or item.get("content") or metadata.get("text")
            else:
                text = getattr(item, "page_content", None)
            if text:
                chunk = str(text)
                if self.max_chunk_chars is not None and self.max_chunk_chars >= 0:
                    chunk = chunk[:self.max_chunk_chars]
                if chunk:
                    contexts.append(chunk)
        if not contexts:
            raise ValueError(
                f"Aucun contexte: search a retourné {len(retrieved)} résultat(s); "
                f"index chargé = {getattr(self.vector_store, 'index', None) is not None}"
            )
        prompt = "Contexte:\n" + "\n\n".join(contexts) + f"\n\nQuestion: {question}"
        answer = _generate_with_retry(self.client, prompt)
        return {"question": question, "answer": answer, "contexts": contexts}


VALID_CATEGORIES = {"simple", "complexe", "bruitée", "hors-périmètre/numérique"}
REQUIRED_QUESTION_COLUMNS = {"question", "ground_truth_expected", "category"}

def _validate_questions_df(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_QUESTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError("Colonnes manquantes dans le CSV des questions : " + ", ".join(sorted(missing)))
    if frame.empty:
        raise ValueError("Le CSV des questions est vide")
    unknown = sorted(set(frame["category"].dropna().astype(str)) - VALID_CATEGORIES)
    if unknown:
        raise ValueError("Catégories inconnues : " + ", ".join(unknown))
    if frame["question"].isna().any() or frame["ground_truth_expected"].isna().any():
        raise ValueError("Les colonnes question et ground_truth_expected ne doivent pas être vides")
    return frame


def _load_questions_df() -> pd.DataFrame:
    """Charge le CSV facultatif ou crée un jeu de questions de repli."""
    global questions_df
    if questions_df is not None:
        return questions_df
    csv_path = project_root / "data/evaluation/questions_metier.csv"
    if csv_path.is_file():
        questions_df = pd.read_csv(csv_path)
        if "question" not in questions_df.columns:
            raise ValueError(f"Colonne 'question' manquante dans {csv_path}")
        if questions_df.empty:
            raise ValueError("Le CSV ne contient aucune ligne")
    else:
        logger.warning("CSV %s introuvable : questions par défaut (sans ground truth).", csv_path)
        questions_df = pd.DataFrame({"question": DEFAULT_QUESTIONS})
    return questions_df


def _ragas_dataset_api():
    """Renvoie les types du schéma moderne si présents, sinon l'API datasets historique."""
    try:
        schema = importlib.import_module("ragas.dataset_schema")
        return schema.EvaluationDataset, schema.SingleTurnSample
    except (ImportError, AttributeError):
        return None, None


def _make_dataset(rows: list[dict[str, Any]], with_ground_truth: bool):
    evaluation_dataset, single_turn_sample = _ragas_dataset_api()
    if evaluation_dataset is not None:
        samples = []
        for row in rows:
            values = {
                "user_input": row["user_input"],
                "response": row["response"],
                "retrieved_contexts": row["retrieved_contexts"],
            }
            if with_ground_truth and row.get("reference") is not None:
                values["reference"] = row["reference"]
            samples.append(single_turn_sample(**values))
        return evaluation_dataset(samples=samples), True

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError("Installez ragas (et datasets pour son API historique).") from exc
    legacy_rows = []
    for row in rows:
        legacy = {"question": row["user_input"], "answer": row["response"],
                  "contexts": row["retrieved_contexts"]}
        if with_ground_truth and row.get("reference") is not None:
            legacy["ground_truth"] = row["reference"]
        legacy_rows.append(legacy)
    return Dataset.from_list(legacy_rows), False


def _call_with_supported_kwargs(cls_or_factory, values: dict[str, Any]):
    """N'injecte que les paramètres pris en charge par la version installée."""
    try:
        signature = inspect.signature(cls_or_factory)
    except (TypeError, ValueError):
        return cls_or_factory()
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                         for p in signature.parameters.values())
    kwargs = {k: v for k, v in values.items()
              if k in signature.parameters or accepts_kwargs}
    return cls_or_factory(**kwargs)


def _make_throttle_callback():
    """Callback LangChain pour freiner chaque appel du juge RAGAS."""
    try:
        from langchain_core.callbacks import BaseCallbackHandler
    except ImportError:
        logger.warning("Callback LangChain indisponible : throttle du juge RAGAS non installable.")
        return None

    class ThrottleCallback(BaseCallbackHandler):
        def on_chat_model_start(self, *args, **kwargs):
            _throttle()

        def on_llm_start(self, *args, **kwargs):
            _throttle()

    return ThrottleCallback()


def _ragas_llm_and_embeddings() -> dict[str, Any]:
    """Construit les wrappers RAGAS LangChain en utilisant explicitement la clé Mistral."""
    try:
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "Évaluation LLM RAGAS : installez langchain-mistralai et une version de ragas "
            "fournissant LangchainLLMWrapper/LangchainEmbeddingsWrapper."
        ) from exc

    api_key = getattr(config, "MISTRAL_API_KEY", None)
    api_key_text = str(api_key).strip() if api_key is not None else ""
    if not api_key_text or api_key_text.lower() in {"test-key", "changeme", "your_api_key"}:
        raise ValueError("MISTRAL_API_KEY est absente ou correspond à une valeur factice.")
    api_key = api_key_text
    model = getattr(config, "MODEL_NAME", "mistral-small-latest")
    embedding_model = getattr(config, "EMBEDDING_MODEL", "mistral-embed")
    # Les intégrations LangChain utilisent l'alias public api_key pour le champ
    # interne mistral_api_key; ne pas fournir les deux noms simultanément.
    chat_args = {"model": model, "temperature": 0, "api_key": api_key}
    throttle_callback = _make_throttle_callback()
    if throttle_callback is not None:
        chat_args["callbacks"] = [throttle_callback]
    embed_args = {"model": embedding_model, "api_key": api_key}
    chat = _call_with_supported_kwargs(ChatMistralAI, chat_args)
    embeddings = _call_with_supported_kwargs(MistralAIEmbeddings, embed_args)
    # Les wrappers historiques acceptent langchain_llm/embeddings sous des
    # noms différents selon la version; leur premier argument positionnel est
    # l'objet LangChain dans les API documentées.
    return {
        "llm": LangchainLLMWrapper(chat),
        "embeddings": LangchainEmbeddingsWrapper(embeddings),
    }


def _resolve_metrics(with_ground_truth: bool, models: dict[str, Any]) -> list[Any]:
    """Choisit les métriques RAGAS disponibles, avec alias modernes et historiques."""
    try:
        metrics_module = importlib.import_module("ragas.metrics")
    except ImportError as exc:
        raise ImportError("ragas est requis : installez-le avant l'évaluation.") from exc
    # Certaines versions modernes n'exposent les classes qu'ici, les anciennes
    # les exposent directement sous ragas.metrics.
    metric_modules = [metrics_module]
    try:
        metric_modules.append(importlib.import_module("ragas.metrics.collections"))
    except ImportError:
        pass

    candidates = [
        (("Faithfulness", "faithfulness"), "llm"),
        (("ResponseRelevancy", "AnswerRelevancy", "answer_relevancy",
          "answer_relevance"), "llm_embeddings"),
    ]
    if with_ground_truth:
        candidates.extend([
            (("LLMContextPrecisionWithReference", "ContextPrecision", "context_precision"), "llm"),
            (("LLMContextRecall", "ContextRecall", "context_recall"), "llm"),
            (("FactualCorrectness", "AnswerCorrectness", "answer_correctness"), "llm"),
        ])
    else:
        candidates.append((
            ("LLMContextPrecisionWithoutReference", "ContextPrecisionWithoutReference"), "llm"
        ))

    metrics = []
    for names, needs in candidates:
        metric = next((getattr(module, name, None)
                       for module in metric_modules for name in names
                       if getattr(module, name, None) is not None), None)
        if metric is None:
            continue
        kwargs = {}
        if "llm" in needs:
            kwargs["llm"] = models["llm"]
        if "embeddings" in needs:
            kwargs["embeddings"] = models["embeddings"]
        if inspect.isclass(metric):
            metric = _call_with_supported_kwargs(metric, kwargs)
        else:
            # Anciennes releases exportent des instances pré-construites.
            for key, value in kwargs.items():
                try:
                    setattr(metric, key, value)
                except (AttributeError, TypeError):
                    logger.warning("Impossible de configurer %s.%s", metric, key)
        metrics.append(metric)
    if not metrics:
        raise RuntimeError("Aucune métrique RAGAS compatible n'a été trouvée.")
    return metrics


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Évaluer le RAG SportSee avec RAGAS.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limiter le nombre de questions après le filtre catégorie.")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Filtrer les catégories (ex. simple complexe) ou liste séparée par virgules.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help="Concurrence RAGAS (défaut : 1 ; le juge reste séquentiel).")
    parser.add_argument("--min-interval", type=float, default=MIN_INTERVAL_S,
                        help="Intervalle minimal entre appels LLM en secondes (défaut : 12).")
    parser.add_argument("--ragas-timeout", type=int, default=DEFAULT_RAGAS_TIMEOUT,
                        help="Timeout par tâche RAGAS en secondes (défaut : 300).")
    parser.add_argument("--ragas-retries", type=int, default=DEFAULT_RAGAS_RETRIES,
                        help="Nombre maximal de tentatives RAGAS (défaut : 10).")
    parser.add_argument("--sleep", type=float, default=RAGAS_QUESTION_DELAY,
                        help="Pause supplémentaire entre générations (défaut : 0 ; throttle global déjà appliqué).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="Nombre de chunks recherchés (défaut : 3).")
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS,
                        help="Longueur maximale de chaque chunk de contexte (défaut : 800).")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit doit être positif ou nul")
    for name in ("max_workers", "ragas_timeout", "ragas_retries", "top_k", "max_chunk_chars"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace(chr(95), chr(45))} doit être positif")
    if args.sleep < 0:
        parser.error("--sleep doit être positif ou nul")
    if args.min_interval < 0:
        parser.error("--min-interval doit être positif ou nul")
    if args.categories:
        args.categories = [part.strip() for item in args.categories
                           for part in item.split(",") if part.strip()]
    return args


def _cache_path() -> Path:
    return project_root / "data/evaluation/cache_reponses.json"


def _load_response_cache(path: Path) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items()
                    if isinstance(v, dict) and v.get("answer") and v.get("contexts")}
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cache réponses illisible (%s) ; un nouveau cache sera créé.", exc)
    return {}


def _save_response_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _jsonl_cache_path() -> Path:
    return project_root / "data/evaluation/answers_cache.jsonl"


def _load_jsonl_response_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    try:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    item = json.loads(line)
                    question = item.get("question")
                    response = item.get("response", item)
                    if question and isinstance(response, dict) and response.get("answer") and response.get("contexts"):
                        cache[str(question)] = response
                except (json.JSONDecodeError, AttributeError, TypeError):
                    logger.warning("Ligne %d ignorée dans le cache JSONL %s.", line_number, path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Cache JSONL illisible (%s) : %s", path, exc)
    return cache


def _append_jsonl_response(path: Path, question: str, response: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"question": question, "response": response}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _scores_frame(score_result) -> pd.DataFrame:
    if hasattr(score_result, "to_pandas"):
        frame = score_result.to_pandas()
    elif isinstance(score_result, pd.DataFrame):
        frame = score_result.copy()
    else:
        try:
            frame = pd.DataFrame(score_result)
        except (TypeError, ValueError):
            frame = pd.DataFrame()
    return frame.reset_index(drop=True)


def _write_reports(report_rows: list[dict[str, Any]], score_result, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    score_frame = _scores_frame(score_result)
    detail = pd.DataFrame(report_rows)
    # RAGAS généralement retourne une ligne par réponse ; si le format varie,
    # conserver les réponses et les scores disponibles sans inventer de valeurs.
    for column in score_frame.columns:
        if column not in detail.columns:
            values = score_frame[column].tolist()
            detail[column] = values[:len(detail)] + [None] * max(0, len(detail) - len(values))
    detail.to_csv(output_dir / "ragas_detail_par_question.csv", index=False, encoding="utf-8-sig")
    score_frame.to_csv(output_dir / "ragas_scores.csv", index=False, encoding="utf-8-sig")

    if "category" in detail.columns:
        excluded = {"question", "category", "answer", "contexts", "ground_truth"}
        numeric_columns = [col for col in detail.columns if col not in excluded
                           and pd.api.types.is_numeric_dtype(detail[col])]
        aggregations = {column: "mean" for column in numeric_columns}
        aggregations["question"] = "count"
        aggregate = detail.groupby("category", dropna=False).agg(aggregations).reset_index()
        aggregate = aggregate.rename(columns={"question": "question_count"})
        aggregate.to_csv(output_dir / "ragas_scores_par_categorie.csv", index=False,
                         encoding="utf-8-sig")
        headers = [str(column) for column in aggregate.columns]
        markdown_lines = ["| " + " | ".join(headers) + " |",
                          "| " + " | ".join("---" for _ in headers) + " |"]
        for values in aggregate.itertuples(index=False, name=None):
            cells = []
            for value in values:
                if pd.isna(value):
                    cells.append("")
                elif isinstance(value, (float, int)) and not isinstance(value, bool):
                    cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
                else:
                    cells.append(str(value).replace("|", "\\\\|"))
            markdown_lines.append("| " + " | ".join(cells) + " |")
        markdown = "\\n".join(markdown_lines)
        (output_dir / "ragas_scores_par_categorie.md").write_text(
            "# Scores RAGAS moyens par catégorie\\n\\n" + markdown + "\\n",
            encoding="utf-8")


def main(argv=None):
    global results, dataset, scores, MIN_INTERVAL_S
    if evaluate is None:
        raise ImportError("ragas est requis : installez-le avant l'évaluation.")
    results = []
    args = _parse_args(argv)
    MIN_INTERVAL_S = max(0.0, float(getattr(args, "min_interval", MIN_INTERVAL_S)))
    df = _load_questions_df().copy()
    if "question" not in df.columns:
        raise ValueError("Colonne 'question' manquante dans les données d'évaluation")
    category_column = "category" if "category" in df.columns else ("categorie" if "categorie" in df.columns else None)
    if category_column and category_column != "category":
        df["category"] = df[category_column]
    elif not category_column:
        df["category"] = "non_catégorisée"
    if args.categories:
        allowed = {value.casefold() for value in args.categories}
        df = df[df["category"].fillna("").astype(str).str.casefold().isin(allowed)]
    if args.limit is not None:
        df = df.head(args.limit)
    has_gt = "ground_truth_expected" in df.columns and df["ground_truth_expected"].notna().any()

    cache_path = _cache_path()
    response_cache = _load_response_cache(cache_path)
    jsonl_path = _jsonl_cache_path()
    # Les caches préexistants sont acceptés ; le cache JSONL est la source la plus récente.
    response_cache.update(_load_jsonl_response_cache(jsonl_path))
    system = _call_with_supported_kwargs(
        RAGSystem, {"top_k": getattr(args, "top_k", DEFAULT_TOP_K),
                    "max_chunk_chars": getattr(args, "max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)}
    )
    valid_rows = [(index, row) for index, row in df.iterrows()
                  if str(row.get("question", "")).strip()
                  and str(row.get("question", "")).strip().lower() != "nan"]
    report_rows = []
    dataset_rows = []
    real_generation_calls = 0
    for _, row in valid_rows:
        question = str(row["question"]).strip()
        category = str(row.get("category", "non_catégorisée"))
        try:
            cached = response_cache.get(question)
            if cached:
                response = cached
                logger.info("Réponse reprise du cache pour %r", question[:60])
            else:
                sleep_seconds = getattr(args, "sleep", RAGAS_QUESTION_DELAY)
                if real_generation_calls > 0 and sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                real_generation_calls += 1
                response = system.query(question)
                answer, contexts = response.get("answer"), response.get("contexts", [])
                if answer and contexts:
                    response_cache[question] = {"answer": str(answer),
                                                "contexts": [str(c) for c in contexts]}
                    _append_jsonl_response(jsonl_path, question, response_cache[question])
                    _save_response_cache(cache_path, response_cache)
            answer, contexts = response.get("answer"), response.get("contexts", [])
        except Exception as exc:
            logger.warning("Question ignorée (%r) : %s: %s",
                           question[:60], type(exc).__name__, exc)
            continue
        if not answer or not contexts:
            logger.warning("Réponse ou contextes vides pour %r", question[:60])
            continue
        max_chunk_chars = getattr(args, "max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
        if max_chunk_chars is not None and max_chunk_chars >= 0:
            contexts = [str(context)[:max_chunk_chars] for context in contexts]
            contexts = [context for context in contexts if context]
            if not contexts:
                logger.warning("Contextes vides après troncature pour %r", question[:60])
                continue
        # Cache, prompt (RAGSystem) et dataset RAGAS utilisent le même découpage.
        response_cache[question] = {"answer": str(answer), "contexts": contexts}
        _save_response_cache(cache_path, response_cache)
        out = {"user_input": question, "response": str(answer),
               "retrieved_contexts": contexts}
        report_row = {"question": question, "category": category,
                      "answer": str(answer), "contexts": json.dumps(contexts, ensure_ascii=False)}
        if has_gt and pd.notna(row.get("ground_truth_expected")):
            gt = str(row["ground_truth_expected"])
            out["reference"] = gt
            report_row["ground_truth"] = gt
        dataset_rows.append(out)
        report_rows.append(report_row)
        results.append({**out, "category": category})

    if not results:
        raise ValueError("Aucune réponse générée")

    dataset, _ = _make_dataset(dataset_rows, has_gt)
    models = _ragas_llm_and_embeddings()
    metrics = _resolve_metrics(has_gt, models)
    eval_kwargs = {"dataset": dataset, "metrics": metrics}
    run_config = None
    try:
        try:
            from ragas.run_config import RunConfig
        except ImportError:
            from ragas import RunConfig
        requested_workers = getattr(args, "max_workers", DEFAULT_MAX_WORKERS)
        if requested_workers != 1:
            logger.warning("--max-workers=%s accepté pour compatibilité ; RAGAS reste séquentiel (1).", requested_workers)
        run_config = _call_with_supported_kwargs(
            RunConfig, {"max_workers": 1,
                        "timeout": 300,
                        "max_retries": 10,
                        "max_wait": 120}
        )
    except ImportError:
        logger.warning("RunConfig indisponible dans cette version de RAGAS.")
    if run_config is not None:
        try:
            params = inspect.signature(evaluate).parameters
            accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                                 for p in params.values())
            if "run_config" in params or accepts_kwargs:
                eval_kwargs["run_config"] = run_config
            else:
                # API ancienne : certaines versions exposent les options directement.
                legacy_options = {"max_workers": 1, "timeout": 300,
                                  "max_retries": 10, "max_wait": 120}
                eval_kwargs.update({key: value for key, value in legacy_options.items()
                                    if key in params})
        except (TypeError, ValueError):
            pass
    try:
        scores = evaluate(**eval_kwargs)
    except Exception as exc:
        if not _is_rate_limit_error(exc) or "run_config" not in eval_kwargs:
            raise
        logger.warning("Limite de débit pendant RAGAS ; nouvelle tentative avec concurrence réduite.")
        try:
            from ragas.run_config import RunConfig
        except ImportError:
            from ragas import RunConfig
        eval_kwargs["run_config"] = _call_with_supported_kwargs(
            RunConfig, {"max_workers": 1, "timeout": 300,
                        "max_retries": 10, "max_wait": 120})
        scores = evaluate(**eval_kwargs)

    _write_reports(report_rows, scores, project_root / "data/evaluation")
    # Compatibilité avec les consommateurs historiques des rapports sous reports/.
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(reports_dir / "ragas_input.csv", index=False)
    _scores_frame(scores).to_csv(reports_dir / "ragas_scores.csv", index=False)
    print("RÉSULTATS RAGAS")
    print(scores)
    return scores


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        main()
    except Exception as exc:
        logger.error("Évaluation RAGAS interrompue : %s: %s", type(exc).__name__, exc)
        sys.exit(1)
