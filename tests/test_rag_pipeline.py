"""Tests des composants Mistral et vector store utilisés par le pipeline RAG."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.llm.mistral_client as mistral_client_module
from src.llm.mistral_client import MistralClient
from src.rag.vector_store import VectorStoreManager


@pytest.mark.unit
def test_mistral_client_generates_answer_with_mocked_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = "Réponse simulée"

    class FakeChat:
        def complete(self, **kwargs):
            assert kwargs["model"] == "mistral-test"
            assert kwargs["messages"][-1] == {"role": "user", "content": "Question ?"}
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=expected))]
            )

    class FakeSDKClient:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.chat = FakeChat()

    monkeypatch.setattr(mistral_client_module, "MistralSDKClient", FakeSDKClient)
    client = MistralClient(api_key="cle-factice", model="mistral-test")

    assert client.generate("Question ?", system_prompt="Réponds brièvement.") == expected


@pytest.mark.unit
def test_rag_components_import_from_src_packages() -> None:
    """Vérifie que les deux composants attendus sont importables."""
    assert MistralClient.__name__ == "MistralClient"
    assert VectorStoreManager.__name__ == "VectorStoreManager"
