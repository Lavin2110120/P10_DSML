"""Client léger pour l'API Mistral utilisé par le projet RAG NBA."""

import logging
from collections.abc import Iterator
from typing import Optional

from mistralai import Mistral as MistralSDKClient


logger = logging.getLogger(__name__)


class MistralClient:
    """Encapsule le client SDK Mistral pour la génération de réponses."""

    def __init__(self, api_key: str, model: str = "mistral-medium-latest"):
        """Initialise le client avec une clé API et un modèle Mistral.

        Args:
            api_key: Clé d'accès à l'API Mistral.
            model: Identifiant du modèle utilisé pour les complétions.

        Raises:
            ValueError: Si la clé API est vide ou ne contient que des espaces.
        """
        if not api_key or not api_key.strip():
            raise ValueError("La clé API Mistral ne peut pas être vide.")

        self.api_key = api_key
        self.model = model
        # Utiliser le client SDK sous un alias évite la récursion avec cette classe.
        self.client = MistralSDKClient(api_key=api_key)

    def generate(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Génère une réponse textuelle à partir d'un prompt utilisateur.

        Args:
            prompt: Message ou question de l'utilisateur.
            system_prompt: Instruction système optionnelle.
            temperature: Température de génération.
            max_tokens: Nombre maximal de tokens générés.

        Returns:
            Le contenu textuel de la première réponse retournée par Mistral.

        Raises:
            RuntimeError: Si l'appel à l'API ou l'extraction de la réponse échoue.
        """
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as exc:
            logger.exception("Erreur lors de la génération avec l'API Mistral")
            raise RuntimeError(
                f"Erreur lors de la génération de la réponse Mistral : {exc}"
            ) from exc

    def generate_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        """Génère une réponse en flux et retourne les fragments de texte.

        Args:
            prompt: Message ou question de l'utilisateur.
            system_prompt: Instruction système optionnelle.
            temperature: Température de génération.
            max_tokens: Nombre maximal de tokens générés.

        Yields:
            Fragments de contenu au fur et à mesure de leur réception.

        Raises:
            RuntimeError: Si l'appel au flux Mistral échoue.
        """
        messages = [
            {"role": "system", "content": system_prompt or ""},
            {"role": "user", "content": prompt},
        ]

        try:
            stream = self.client.chat.stream(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for event in stream:
                choices = getattr(event.data, "choices", [])
                if not choices:
                    continue
                content = getattr(choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as exc:
            logger.exception("Erreur lors du streaming avec l'API Mistral")
            raise RuntimeError(
                f"Erreur lors du streaming de la réponse Mistral : {exc}"
            ) from exc
