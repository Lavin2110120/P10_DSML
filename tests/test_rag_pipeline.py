"""Tests du pipeline RAG : client Mistral (SDK mocké) et imports des composants."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.mistral_client import MistralClient
from src.rag.vector_store import VectorStoreManager


@pytest.mark.unit
def test_mistral_client_generates_answer_with_mocked_sdk() -> None:
    expected = "Réponse simulée"

    class FakeChat:
        def complete(self, **kwargs):
            # generate() appelle self.client.chat.complete(model=..., messages=..., ...)
            assert kwargs["model"] == "mistral-test"
            # Avec un system_prompt, messages = [system, user] : le dernier est le user
            assert kwargs["messages"][-1] == {"role": "user", "content": "Question ?"}
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=expected))]
            )

    class FakeSDKClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    # On contourne __init__ (qui instancierait le vrai SDK Mistral) :
    # aucun monkeypatch sur un nom importé -> plus d'AttributeError/NameError.
    client = MistralClient.__new__(MistralClient)
    client.api_key = "cle-factice"
    client.model = "mistral-test"
    client.client = FakeSDKClient()

    assert client.generate("Question ?", system_prompt="Réponds brièvement.") == expected


@pytest.mark.unit
def test_rag_components_import_from_src_packages() -> None:
    """Vérifie que les deux composants attendus sont importables."""
    assert MistralClient.__name__ == "MistralClient"
    assert VectorStoreManager.__name__ == "VectorStoreManager"