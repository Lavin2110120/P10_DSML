"""Structured RAG generation with optional Pydantic AI and Gemini fallback."""
from __future__ import annotations
import logging
from typing import Any
from src import config
from src.models import Chunk, RAGAnswer

_LOG = logging.getLogger(__name__)
SYSTEM_PROMPT = (
    "Réponds en français à partir du contexte fourni. Ne crée pas de faits ni de sources. "
    "Retourne answer, sources (chunks cités), confidence entre 0 et 1, grounded et abstained. "
    "Si le contexte est insuffisant, indique-le, grounded=false et abstained=true."
)

def _source_chunks(sources: list[dict[str, Any]] | None) -> list[Chunk]:
    result=[]
    for i, source in enumerate(sources or []):
        try:
            value=dict(source)
            value.setdefault("chunk_index", i)
            result.append(Chunk.model_validate(value))
        except Exception as exc:
            _LOG.warning("Source RAG %d ignorée : %s", i, exc)
    return result

def generate_structured_answer(prompt: str, context: str, sources: list[dict[str, Any]], client: Any) -> RAGAnswer:
    """Try Pydantic AI; fall back safely to GeminiClient.generate_content."""
    message=f"Question : {prompt}\n\nContexte :\n{context}"
    try:
        from pydantic_ai import Agent
        model_name=getattr(config, "GEMINI_MODEL", getattr(config, "MODEL_NAME", "gemini-2.0-flash"))
        api_key=getattr(config, "GOOGLE_API_KEY", None) or getattr(config, "GEMINI_API_KEY", None)
        agent=Agent(f"google-gla:{model_name}", output_type=RAGAnswer, instructions=SYSTEM_PROMPT, deps_type=type(None))
        # Explicitly pass config key when supported by the provider through env only
        # (config has loaded the .env; never expose it to logs).
        if api_key:
            import os
            os.environ["GOOGLE_API_KEY"] = api_key
        result=agent.run_sync(message)
        output=getattr(result, "output", getattr(result, "data", None))
        return RAGAnswer.model_validate(output)
    except Exception as exc:
        _LOG.info("Pydantic AI indisponible ou génération structurée échouée; repli Gemini (%s)", type(exc).__name__)
    raw=client.generate_content(message, system_prompt=SYSTEM_PROMPT)
    valid_sources=_source_chunks(sources)
    if not context.strip() or not valid_sources:
        return RAGAnswer(answer=str(raw).strip() or "Je ne dispose pas d’informations suffisantes dans le contexte.", sources=[], grounded=False, abstained=True)
    # The fallback cannot assert source selection; attach only retrieved, validated evidence.
    return RAGAnswer(answer=str(raw).strip() or "Je ne dispose pas d’informations suffisantes dans le contexte.", sources=valid_sources, grounded=True, abstained=False)
