"""Évaluation RAGAS du pipeline RAG Gemini (``python -m src.evaluation.evaluate_ragas``)."""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import logging
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

# Important: apply the project's TLS workaround before importing any Google,
# LangChain, RAGAS, or other network-capable library.
from src.ssl_bypass import disable_ssl_verification

disable_ssl_verification()

import pandas as pd

try:
    from ragas import evaluate
except ImportError:
    evaluate = None

from src import config
from src.rag.vector_store import VectorStoreManager
from src.llm.gemini_client import GeminiClient
from src.llm.structured_rag import generate_structured_answer
from src.models import RAGAnswer
from src.observability import configure, span

logger = logging.getLogger(__name__)
project_root = Path(config.PROJECT_ROOT)

DEFAULT_QUESTIONS = [
    "Quels sont les principaux sujets traités dans les documents ?",
    "Résumez le contenu principal des documents disponibles.",
    "Quelles informations clés peut-on retenir de ces documents ?",
]

# Globals kept for compatibility with downstream scripts and test fixtures.
questions_df: pd.DataFrame | None = None
results: list[dict[str, Any]] = []
dataset = None
scores = None

MIN_INTERVAL_S = max(0.0, float(os.getenv("MIN_INTERVAL_S", "0")))
GENERATION_MAX_RETRIES = max(0, int(os.getenv("RAGAS_GENERATION_RETRIES", "5")))
GENERATION_RETRY_BASE_DELAY = max(0.0, float(os.getenv("RAGAS_RETRY_BASE_DELAY", "2")))
GENERATION_RETRY_MAX_DELAY = max(0.0, float(os.getenv("RAGAS_RETRY_MAX_DELAY", "30")))
DEFAULT_MAX_WORKERS = 1
DEFAULT_RAGAS_TIMEOUT = 120
DEFAULT_RAGAS_RETRIES = 5
DEFAULT_RAGAS_MAX_WAIT = 120
RAGAS_QUESTION_DELAY = 0.0
DEFAULT_TOP_K = int(getattr(config, "SEARCH_K", 3))
DEFAULT_MAX_CHUNK_CHARS = 800
_throttle_lock = threading.Lock()
_last_call_time: float | None = None


def _throttle(interval: float | None = None) -> None:
    """Optionally serialize and space calls made by this process."""
    global _last_call_time
    minimum = MIN_INTERVAL_S if interval is None else max(0.0, float(interval))
    with _throttle_lock:
        now = time.monotonic()
        wait = 0.0 if _last_call_time is None else minimum - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
            now = time.monotonic()
        _last_call_time = now


def _http_status(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    status = (getattr(exc, "status_code", None) or getattr(exc, "status", None)
              or getattr(response, "status_code", None))
    if status is None:
        code = getattr(exc, "code", None)
        if callable(code):
            try:
                code = code()
            except Exception:
                code = None
        status = code
    return str(status) if status is not None else None


def _is_retryable_error(exc: Exception) -> bool:
    """Identify transient Gemini HTTP 429/503 (including wrapped SDK errors)."""
    message = str(exc).lower()
    if _http_status(exc) in {"429", "503", "statuscode.too_many_requests", "statuscode.unavailable"}:
        return True
    return bool(re.search(r"\b(?:429|503)\b", message) or any(token in message for token in (
        "resource_exhausted", "too many requests", "rate limit", "rate_limited",
        "service unavailable", "temporarily unavailable", "high demand",
    )))


# Historical helper name retained for test and caller compatibility.
_is_rate_limit_error = _is_retryable_error


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            return max(0.0, (parsedate_to_datetime(str(raw)) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _generate_with_retry(client: Any, prompt: str) -> str:
    """Call the real project's GeminiClient, retrying transient 429/503 failures."""
    for attempt in range(GENERATION_MAX_RETRIES + 1):
        _throttle()
        try:
            generate = getattr(client, "generate_content", None)
            if generate is None:  # backward-compatible mock/client interface
                generate = getattr(client, "generate", None) or getattr(client, "generate_response")
            try:
                answer = generate(prompt=prompt)
            except TypeError as exc:
                # Older mock clients may accept positional prompt only.
                if "prompt" not in str(exc) and "keyword" not in str(exc):
                    raise
                answer = generate(prompt)
            if isinstance(answer, str):
                return answer.strip()
            return str(answer).strip() if answer is not None else ""
        except Exception as exc:
            if not _is_retryable_error(exc) or attempt >= GENERATION_MAX_RETRIES:
                raise
            exponential = min(GENERATION_RETRY_BASE_DELAY * (2 ** attempt), GENERATION_RETRY_MAX_DELAY)
            wait = _retry_after_seconds(exc)
            delay = min(wait if wait is not None else exponential, max(GENERATION_RETRY_MAX_DELAY, 1.0))
            delay += random.uniform(0, min(1.0, GENERATION_RETRY_BASE_DELAY))
            logger.warning("Gemini %s transitoire; nouvel essai %d/%d dans %.1f s.",
                           _http_status(exc) or "429/503", attempt + 1,
                           GENERATION_MAX_RETRIES, delay)
            time.sleep(delay)
    return ""  # loop always returns or raises


class RAGSystem:
    """Adapter on the project's structured RAG, validated response and chunk search."""
    def __init__(self, top_k: int = DEFAULT_TOP_K, max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS):
        self.top_k = top_k
        self.max_chunk_chars = max_chunk_chars
        self.client = GeminiClient()
        self.vector_store = VectorStoreManager()
        try:
            self.vector_store.load()
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                f"Index FAISS absent ou de dimension incohérente ({exc}). "
                "Construisez/réindexez le vector store avec `make vectorstore`."
            ) from exc
        index = getattr(self.vector_store, "index", None)
        expected_dim = int(getattr(config, "EMBEDDING_DIM"))
        actual_dim = getattr(index, "d", None)
        if actual_dim is None or int(actual_dim) != expected_dim:
            raise RuntimeError(
                f"Dimension d'index FAISS incohérente : {actual_dim}, attendu EMBEDDING_DIM={expected_dim}. "
                "Reconstruisez l'index avec `make vectorstore`."
            )

    def query(self, question: str) -> dict[str, Any]:
        retrieved = self.vector_store.search(question, top_k=self.top_k)
        chunks: list[dict[str, Any]] = []
        contexts: list[str] = []
        for item in retrieved or []:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("page_content")
                chunk = dict(item)
            else:
                text = getattr(item, "text", None) or getattr(item, "page_content", None)
                chunk = {"text": text, "metadata": getattr(item, "metadata", {}) or {}}
            if text is None:
                continue
            text = str(text).strip()
            if self.max_chunk_chars is not None and self.max_chunk_chars >= 0:
                text = text[:self.max_chunk_chars]
            if text:
                chunks.append(chunk)
                contexts.append(text)
        if not contexts:
            raise ValueError("Aucun chunk non vide trouvé dans les résultats de recherche FAISS.")
        context = "\n\n---\n\n".join(contexts)
        mode = "structured"
        try:
            answer_obj = generate_structured_answer(question, context, chunks, self.client)
            answer_obj = RAGAnswer.model_validate(answer_obj)
        except Exception:
            # Même fallback direct que GeminiChat.py, avec contexte et validation Pydantic.
            mode = "fallback"
            raw_answer = self.client.generate_content(
                question,
                system_prompt=(
                    "Réponds en français uniquement à partir du contexte fourni. "
                    "Si l'information manque, dis-le clairement.\n\n"
                    f"CONTEXTE :\n{context}"
                ),
            )
            answer_obj = RAGAnswer(
                answer=str(raw_answer).strip() or "Je ne dispose pas d’informations suffisantes dans le contexte.",
                sources=[], grounded=False, abstained=True,
            )
        if not answer_obj.answer.strip():
            raise ValueError("La réponse générée est vide.")
        return {"question": question, "answer": answer_obj.answer,
                "contexts": contexts, "mode": mode}


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
    """Charge et valide le jeu métier requis; l'absence n'est jamais masquée."""
    global questions_df
    if questions_df is not None:
        return _validate_questions_df(questions_df.copy())
    csv_path = project_root / "data/evaluation/questions_metier.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Fichier de questions métier absent : {csv_path}. "
            "Ajoutez data/evaluation/questions_metier.csv avant l'évaluation."
        )
    questions_df = pd.read_csv(csv_path)
    return _validate_questions_df(questions_df.copy())


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
    """Build Gemini judge wrappers with REST transport to avoid the gRPC TLS path."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "Installez langchain-google-genai (>=2,<3) et ragas pour le juge Gemini."
        ) from exc

    api_key = (getattr(config, "GOOGLE_API_KEY", None)
               or getattr(config, "GEMINI_API_KEY", None)
               or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    if not api_key or str(api_key).strip().lower() in {"test-key", "changeme", "your_api_key"}:
        raise ValueError("La clé Gemini (GOOGLE_API_KEY ou GEMINI_API_KEY) est absente/incorrecte.")
    model = (getattr(config, "GEMINI_MODEL", None)
             or getattr(config, "MODEL_NAME", None) or "gemini-2.0-flash")
    embedding_model = (getattr(config, "GEMINI_EMBEDDING_MODEL", None)
                       or getattr(config, "EMBEDDING_MODEL_NAME", None)
                       or "models/text-embedding-004")
    chat_args = {"model": model, "temperature": 0, "transport": "rest"}
    # LangChain Gemini 2.x uses api_key for chat and google_api_key for embeddings.
    chat_args.update({"api_key": api_key, "google_api_key": api_key})
    embed_args = {"model": embedding_model, "transport": "rest",
                  "google_api_key": api_key}
    try:
        chat_supports_transport = "transport" in inspect.signature(ChatGoogleGenerativeAI).parameters
        embeddings_support_transport = "transport" in inspect.signature(GoogleGenerativeAIEmbeddings).parameters
    except (TypeError, ValueError):
        chat_supports_transport = embeddings_support_transport = True
    if not (chat_supports_transport and embeddings_support_transport):
        logger.warning(
            "Cette version de langchain-google-genai n’accepte pas transport='rest' "
            "sur les deux wrappers; repli documenté vers le transport par défaut."
        )
        chat_args.pop("transport", None)
        embed_args.pop("transport", None)
    try:
        chat = _call_with_supported_kwargs(ChatGoogleGenerativeAI, chat_args)
        embeddings = _call_with_supported_kwargs(GoogleGenerativeAIEmbeddings, embed_args)
    except (TypeError, ValueError) as exc:
        # Compatibility fallback if an integration version exposes transport
        # in its signature but rejects it at runtime.
        if "transport" not in chat_args and "transport" not in embed_args:
            raise
        logger.warning(
            "Paramètre transport='rest' refusé (%s); repli documenté vers le transport par défaut.",
            exc,
        )
        chat_args.pop("transport", None)
        embed_args.pop("transport", None)
        chat = _call_with_supported_kwargs(ChatGoogleGenerativeAI, chat_args)
        embeddings = _call_with_supported_kwargs(GoogleGenerativeAIEmbeddings, embed_args)
    throttle_callback = _make_throttle_callback()
    if throttle_callback is not None:
        try:
            chat.callbacks = [throttle_callback]
        except Exception:
            logger.debug("Callback de temporisation indisponible pour le modèle Gemini.")
    return {"llm": LangchainLLMWrapper(chat),
            "embeddings": LangchainEmbeddingsWrapper(embeddings)}


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
                        help="Intervalle minimal entre appels LLM en secondes (défaut : 0).")
    parser.add_argument("--ragas-timeout", type=int, default=DEFAULT_RAGAS_TIMEOUT,
                        help="Timeout par tâche RAGAS en secondes (défaut : 120).")
    parser.add_argument("--ragas-retries", type=int, default=DEFAULT_RAGAS_RETRIES,
                        help="Nombre maximal de tentatives RAGAS (défaut : 5).")
    raise_default = os.getenv("RAGAS_RAISE_EXCEPTIONS", "true").strip().lower() not in {"0", "false", "no", "off"}
    raise_group = parser.add_mutually_exclusive_group()
    raise_group.add_argument("--raise-exceptions", dest="raise_exceptions", action="store_true")
    raise_group.add_argument("--no-raise-exceptions", dest="raise_exceptions", action="store_false")
    parser.set_defaults(raise_exceptions=raise_default)
    parser.add_argument("--sleep", type=float, default=RAGAS_QUESTION_DELAY,
                        help="Pause supplémentaire entre générations (défaut : 0 ; throttle optionnel).")
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

        # Tableau croisé catégories × métriques (moyennes), conservé également en CSV.
        if numeric_columns:
            cross = detail.pivot_table(
                index="category", values=numeric_columns, aggfunc="mean", dropna=False
            )
            cross.to_csv(output_dir / "ragas_scores_croises_categorie_metrique.csv",
                         encoding="utf-8-sig")
            cross_headers = ["category", *[str(column) for column in cross.columns]]
            cross_lines = ["| " + " | ".join(cross_headers) + " |",
                           "| " + " | ".join("---" for _ in cross_headers) + " |"]
            for category_value, values in cross.iterrows():
                cells = [str(category_value)]
                cells.extend("" if pd.isna(value) else f"{float(value):.4f}" for value in values)
                cross_lines.append("| " + " | ".join(cells) + " |")
            (output_dir / "ragas_scores_croises_categorie_metrique.md").write_text(
                "# Tableau croisé : catégories × métriques (moyennes)\n\n"
                + "\n".join(cross_lines) + "\n", encoding="utf-8")


def main(argv=None):
    global results, dataset, scores, MIN_INTERVAL_S, questions_df
    if evaluate is None:
        raise ImportError("ragas est requis : installez-le avant l'évaluation.")
    results = []
    configure()
    args = _parse_args(argv)
    MIN_INTERVAL_S = max(0.0, float(getattr(args, "min_interval", MIN_INTERVAL_S)))
    # Accepter l'ancien alias français, puis toujours appliquer le validateur métier.
    source_df = questions_df.copy() if questions_df is not None else None
    if source_df is None:
        csv_path = project_root / "data/evaluation/questions_metier.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"Fichier de questions métier absent : {csv_path}. "
                "Ajoutez data/evaluation/questions_metier.csv avant l'évaluation."
            )
        source_df = pd.read_csv(csv_path)
        questions_df = source_df.copy()
    if "category" not in source_df.columns and "categorie" in source_df.columns:
        source_df["category"] = source_df["categorie"]
    df = _validate_questions_df(source_df.copy())
    if args.categories:
        allowed = {value.casefold() for value in args.categories}
        df = df[df["category"].fillna("").astype(str).str.casefold().isin(allowed)]
    if args.limit is not None:
        df = df.head(args.limit)
    has_gt = ("ground_truth_expected" in df.columns and not df.empty
              and df["ground_truth_expected"].notna().all()
              and df["ground_truth_expected"].astype(str).str.strip().ne("").all())
    if "ground_truth_expected" in df.columns and df["ground_truth_expected"].notna().any() and not has_gt:
        logger.warning("Ground truth partielle : métriques avec référence désactivées pour éviter des échantillons incomplets.")

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
    for question_index, row in valid_rows:
        question = str(row["question"]).strip()
        category = str(row.get("category", "non_catégorisée"))
        question_id = row.get("id", row.get("question_id", question_index))
        with span("ragas.question", category=category, question_id=str(question_id)):
            try:
                cached = response_cache.get(question)
                # Anciennes entrées n'ont pas de mode; les régénérer plutôt que d'inventer un mode.
                if cached and cached.get("mode") in {"structured", "fallback"}:
                    response = cached
                    logger.info("Réponse reprise du cache pour l'identifiant %s", question_id)
                else:
                    sleep_seconds = getattr(args, "sleep", RAGAS_QUESTION_DELAY)
                    if real_generation_calls > 0 and sleep_seconds > 0:
                        time.sleep(sleep_seconds)
                    real_generation_calls += 1
                    response = system.query(question)
                    answer, contexts = response.get("answer"), response.get("contexts", [])
                    if answer and contexts:
                        response_cache[question] = {
                            "answer": str(answer), "contexts": [str(c) for c in contexts],
                            "mode": response.get("mode", "structured"),
                        }
                        _append_jsonl_response(jsonl_path, question, response_cache[question])
                        _save_response_cache(cache_path, response_cache)
                answer, contexts = response.get("answer"), response.get("contexts", [])
                mode = response.get("mode", "structured")
                if mode not in {"structured", "fallback"}:
                    mode = "structured"
            except Exception as exc:
                logger.warning("Question d'identifiant %s ignorée : %s: %s",
                               question_id, type(exc).__name__, exc)
                continue
        answer = str(answer).strip() if answer is not None else ""
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [str(context).strip() for context in (contexts or []) if context is not None]
        contexts = [context for context in contexts if context]
        if not answer or not contexts:
            logger.warning("Réponse ou contextes vides pour %r", question[:60])
            continue
        max_chunk_chars = getattr(args, "max_chunk_chars", DEFAULT_MAX_CHUNK_CHARS)
        if max_chunk_chars is not None and max_chunk_chars >= 0:
            contexts = [str(context)[:max_chunk_chars].strip() for context in contexts]
            contexts = [context for context in contexts if context]
            if not contexts:
                logger.warning("Contextes vides après troncature pour %r", question[:60])
                continue
        # Cache, prompt (RAGSystem) et dataset RAGAS utilisent le même découpage.
        response_cache[question] = {"answer": str(answer), "contexts": contexts, "mode": mode}
        _save_response_cache(cache_path, response_cache)
        out = {"user_input": question, "response": str(answer),
               "retrieved_contexts": contexts}
        report_row = {"question": question, "category": category, "mode": mode,
                      "answer": str(answer), "contexts": json.dumps(contexts, ensure_ascii=False)}
        if has_gt and pd.notna(row.get("ground_truth_expected")):
            gt = str(row["ground_truth_expected"])
            out["reference"] = gt
            report_row["ground_truth"] = gt
        dataset_rows.append(out)
        report_rows.append(report_row)
        results.append({**out, "category": category, "mode": mode})

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
                        "timeout": getattr(args, "ragas_timeout", DEFAULT_RAGAS_TIMEOUT),
                        "max_retries": getattr(args, "ragas_retries", DEFAULT_RAGAS_RETRIES),
                        "max_wait": DEFAULT_RAGAS_MAX_WAIT}
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
                legacy_options = {
                    "max_workers": 1,
                    "timeout": getattr(args, "ragas_timeout", DEFAULT_RAGAS_TIMEOUT),
                    "max_retries": getattr(args, "ragas_retries", DEFAULT_RAGAS_RETRIES),
                    "max_wait": DEFAULT_RAGAS_MAX_WAIT,
                }
                eval_kwargs.update({key: value for key, value in legacy_options.items()
                                    if key in params})
        except (TypeError, ValueError):
            pass
    eval_kwargs["raise_exceptions"] = bool(getattr(args, "raise_exceptions", True))
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
            RunConfig, {"max_workers": 1,
                        "timeout": getattr(args, "ragas_timeout", DEFAULT_RAGAS_TIMEOUT),
                        "max_retries": getattr(args, "ragas_retries", DEFAULT_RAGAS_RETRIES),
                        "max_wait": DEFAULT_RAGAS_MAX_WAIT})
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
