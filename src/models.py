"""Prototype de contrats Pydantic v2 pour les documents et réponses RAG.

Ces modèles gardent le schéma de chargement actuel (text/source/page/metadata)
compatible, avec quelques alias d'entrée historiques tolérés.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Document(BaseModel):
    """Document issu du chargeur; les champs additionnels du corpus sont préservés."""

    model_config = ConfigDict(extra="allow")

    text: str = Field(min_length=1, description="Texte canonique du document")
    source: str | None = None
    page: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_legacy_text_and_metadata(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        value = dict(value)
        if "text" not in value:
            for alias in ("page_content", "content"):
                if value.get(alias) is not None:
                    value["text"] = value[alias]
                    break
        metadata = value.get("metadata")
        if isinstance(metadata, dict):
            value.setdefault("source", metadata.get("source"))
            value.setdefault("page", metadata.get("page"))
        return value

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text ne peut pas être vide")
        return value

    @model_validator(mode="after")
    def synchronize_source_metadata(self) -> "Document":
        if self.source is not None:
            self.metadata.setdefault("source", self.source)
        if self.page is not None:
            self.metadata.setdefault("page", self.page)
        self.metadata = {key: value for key, value in self.metadata.items() if value is not None}
        return self


class Chunk(Document):
    """Segment indexable; chunk_index suit le schéma du chargeur courant."""

    chunk_index: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def accept_chunk_index_in_metadata(cls, value: Any) -> Any:
        if isinstance(value, dict):
            value = dict(value)
            metadata = value.get("metadata")
            if "chunk_index" not in value and isinstance(metadata, dict):
                if "chunk_index" in metadata:
                    value["chunk_index"] = metadata["chunk_index"]
        return value

    @model_validator(mode="after")
    def synchronize_chunk_metadata(self) -> "Chunk":
        self.metadata.setdefault("chunk_index", self.chunk_index)
        return self


class RAGAnswer(BaseModel):
    """Réponse vérifiable avec références aux chunks récupérés."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    sources: list[Chunk] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    grounded: bool = True
    abstained: bool = False

    @field_validator("answer")
    @classmethod
    def answer_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("answer ne peut pas être vide")
        return value

    @model_validator(mode="after")
    def enforce_grounding_consistency(self) -> "RAGAnswer":
        if self.grounded and not self.sources:
            raise ValueError("une réponse grounded doit citer au moins une source")
        if self.abstained and self.grounded:
            raise ValueError("une réponse abstained ne peut pas être grounded")
        return self
