"""Google Gemini client for text generation and vector-store embeddings."""

from __future__ import annotations

import logging
import os
import math
from collections.abc import Iterable, Iterator
from typing import Any

from src import config
from src.ssl_bypass import disable_ssl_verification
from src.observability import span

# Le contournement SSL reste désactivé sauf opt-in explicite dans la configuration.
if os.getenv("DISABLE_SSL_VERIFY", "0") == "1":
    disable_ssl_verification()

_LOG = logging.getLogger(__name__)


def _sanitize_text(text: Any) -> str:
    """Nettoie et convertit l'entrée en chaîne de caractères non vide."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def _normalize_task_type(task_type: str | None) -> str | None:
    """Normalise le nom du task_type pour l'API Gemini."""
    if not task_type:
        return None
    normalized = str(task_type).strip().lower().replace("-", "_")
    mapping = {
        "retrieval_document": "RETRIEVAL_DOCUMENT",
        "retrieval_doc": "RETRIEVAL_DOCUMENT",
        "retrieval_query": "RETRIEVAL_QUERY",
        "semantic_similarity": "SEMANTIC_SIMILARITY",
        "classification": "CLASSIFICATION",
        "clustering": "CLUSTERING",
        "question_answering": "QUESTION_ANSWERING",
        "fact_verification": "FACT_VERIFICATION",
    }
    return mapping.get(normalized, str(task_type).upper())


def _extract_embedding_vector(entry: Any) -> list[float]:
    """Extrait un vecteur de floats depuis la réponse du SDK Google."""
    if isinstance(entry, (list, tuple)):
        return [float(x) for x in entry]
    if hasattr(entry, "values"):
        vals = entry.values
        if hasattr(vals, "to_py"):
            vals = vals.to_py()
        return [float(x) for x in vals]
    if isinstance(entry, dict):
        for key in ("embedding", "values"):
            if key in entry:
                return _extract_embedding_vector(entry[key])
    raise TypeError(f"Format d'embedding inconnu : {type(entry)!r}")


class GeminiClient:
    """Client unifié pour Gemini (génération et embeddings)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        self.api_key = (
            api_key
            or getattr(config, "GEMINI_API_KEY", None)
            or getattr(config, "GOOGLE_API_KEY", None)
        )
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY introuvable dans la configuration ou l'environnement.")

        self.model_name = (
            model_name
            or getattr(config, "GEMINI_MODEL", None)
            or getattr(config, "GENERATION_MODEL", None)
            or "gemini-1.5-flash"
        )
        self.embedding_model_name = (
            embedding_model_name
            or getattr(config, "GEMINI_EMBEDDING_MODEL", None)
            or getattr(config, "EMBEDDING_MODEL", None)
            or "text-embedding-004"
        )

        try:
            from google import genai  # type: ignore

            self._backend = "google-genai"
            self._client = genai.Client(api_key=self.api_key)
        except ImportError:
            try:
                import google.generativeai as gai  # type: ignore

                gai.configure(api_key=self.api_key)
                self._backend = "google-generativeai"
                self._client = gai
            except ImportError as exc:
                raise ImportError(
                    "Installez le SDK Google : pip install google-genai "
                    "(ou google-generativeai)."
                ) from exc

    # --------------------------------------------------------------------------
    # Génération de texte
    # --------------------------------------------------------------------------
    def generate_content(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> str:
        """Génère du texte via Gemini."""
        with span("rag.gemini_generate", model=self.model_name, prompt_length=len(prompt or "")):
            return self._generate_content_uninstrumented(prompt, system_prompt, temperature)

    def _generate_content_uninstrumented(self, prompt: str, system_prompt: str | None, temperature: float) -> str:
        if self._backend == "google-genai":
            config_kwargs: dict[str, Any] = {"temperature": temperature}
            if system_prompt:
                config_kwargs["system_instruction"] = system_prompt
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_kwargs,
            )
            return response.text or ""

        kwargs: dict[str, Any] = {}
        if system_prompt:
            kwargs["system_instruction"] = system_prompt
        model = self._client.GenerativeModel(self.model_name, **kwargs)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return response.text or ""


    def stream_content(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.2,
    ) -> Iterator[str]:
        """Génère du texte en streaming, avec une trace dépourvue du prompt."""
        with span("rag.gemini_generate_stream", model=self.model_name, prompt_length=len(prompt or "")):
            if self._backend == "google-genai":
                config_kwargs: dict[str, Any] = {"temperature": temperature}
                if system_prompt:
                    config_kwargs["system_instruction"] = system_prompt
                for chunk in self._client.models.generate_content_stream(
                    model=self.model_name, contents=prompt, config=config_kwargs
                ):
                    if chunk.text:
                        yield chunk.text
                return

            kwargs: dict[str, Any] = {}
            if system_prompt:
                kwargs["system_instruction"] = system_prompt
            model = self._client.GenerativeModel(self.model_name, **kwargs)
            for chunk in model.generate_content(
                prompt, generation_config={"temperature": temperature}, stream=True
            ):
                if chunk.text:
                    yield chunk.text

    # --------------------------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------------------------
    def get_embedding(
        self,
        text: str,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
        **kwargs: Any,
    ) -> list[float]:
        """Calcule l'embedding pour un texte unique."""
        clean = _sanitize_text(text) or " "
        embeddings = self.get_embeddings(
            [clean],
            task_type=task_type,
            output_dimensionality=output_dimensionality,
            **kwargs,
        )
        return embeddings[0]

    def get_embeddings(
        self,
        texts: Iterable[str],
        batch_size: int = 16,
        task_type: str | None = None,
        output_dimensionality: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Calcule par lots les embeddings d'une liste de textes."""
        prepared = [_sanitize_text(t) or " " for t in texts]
        with span("rag.gemini_embeddings", model=self.embedding_model_name, input_count=len(prepared)):
            return self._get_embeddings_uninstrumented(prepared, batch_size, task_type, output_dimensionality, **kwargs)

    def _get_embeddings_uninstrumented(self, prepared: list[str], batch_size: int, task_type: str | None, output_dimensionality: int | None, **kwargs: Any) -> list[list[float]]:
        if not prepared:
            return []

        norm_task_type = _normalize_task_type(task_type)
        all_vectors: list[list[float]] = []

        if self._backend == "google-genai":
            from google.genai import types  # type: ignore

            config_params: dict[str, Any] = {}
            if norm_task_type:
                config_params["task_type"] = norm_task_type
            if output_dimensionality:
                config_params["output_dimensionality"] = output_dimensionality

            cfg = types.EmbedContentConfig(**config_params) if config_params else None

            for i in range(0, len(prepared), batch_size):
                batch = prepared[i : i + batch_size]
                call_kwargs: dict[str, Any] = {
                    "model": self.embedding_model_name,
                    "contents": batch,
                }
                if cfg is not None:
                    call_kwargs["config"] = cfg

                with span("rag.gemini_embedding_batch", model=self.embedding_model_name, input_count=len(batch)):
                    response = self._client.models.embed_content(**call_kwargs)
                embeddings_data = getattr(response, "embeddings", None) or []
                for entry in embeddings_data:
                    all_vectors.append(_extract_embedding_vector(entry))
            return all_vectors

        # Backend legacy google-generativeai
        model_id = (
            self.embedding_model_name
            if self.embedding_model_name.startswith("models/")
            else f"models/{self.embedding_model_name}"
        )

        for i in range(0, len(prepared), batch_size):
            batch = prepared[i : i + batch_size]
            legacy_kwargs: dict[str, Any] = {
                "model": model_id,
                "content": batch,
            }
            if norm_task_type:
                legacy_kwargs["task_type"] = norm_task_type
            if output_dimensionality:
                legacy_kwargs["output_dimensionality"] = output_dimensionality

            with span("rag.gemini_embedding_batch", model=self.embedding_model_name, input_count=len(batch)):
                result = self._client.embed_content(**legacy_kwargs)
            raw = result.get("embedding", [])
            if raw and isinstance(raw[0], (int, float)):
                all_vectors.append([float(x) for x in raw])
            else:
                for entry in raw:
                    all_vectors.append(_extract_embedding_vector(entry))

        return all_vectors