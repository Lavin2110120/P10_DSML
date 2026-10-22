"""
Client Mistral AI - Compatible avec toutes les versions du SDK (v0.x et v1.x+)
"""
import logging
import os
from collections.abc import Iterator
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# DÉTECTION AUTOMATIQUE DE LA VERSION DU SDK
# ============================================================
MISTRAL_SDK_VERSION = None
MistralSDKClient = None
USE_LEGACY_API = False

try:
    # Tentative 1 : Nouvelle API (v1.x+) - classe Mistral à la racine
    from mistralai import Mistral as MistralSDKClient
    MISTRAL_SDK_VERSION = "1.x+"
    USE_LEGACY_API = False
    logger.info("✅ SDK Mistral v1.x+ détecté (classe 'Mistral')")
except ImportError:
    try:
        # Tentative 2 : Ancienne API (v0.x) - MistralClient dans client.py
        from mistralai.client import MistralClient as MistralSDKClient
        MISTRAL_SDK_VERSION = "0.x"
        USE_LEGACY_API = True
        logger.info("✅ SDK Mistral v0.x détecté (classe 'MistralClient' depuis client.py)")
    except ImportError:
        try:
            # Tentative 3 : Ancienne API (v0.x) - MistralClient à la racine
            from mistralai import MistralClient as MistralSDKClient
            MISTRAL_SDK_VERSION = "0.x"
            USE_LEGACY_API = True
            logger.info("✅ SDK Mistral v0.x détecté (classe 'MistralClient' à la racine)")
        except ImportError:
            raise ImportError(
                "❌ Impossible d'importer le client Mistral.\n"
                "Installez le package avec :\n"
                "  pip install mistralai          # v1.x+ (recommandé)\n"
                "  pip install 'mistralai<1.0'    # v0.x (legacy)"
            )

# ============================================================
# CLASSE WRAPPER UNIFIÉE
# ============================================================
class MistralClient:
    """
    Wrapper unifié pour le client Mistral, compatible v0.x et v1.x+
    """

    def __init__(self, api_key: str = None, model: str = "mistral-medium-latest"):
        """
        Initialise le client Mistral.

        Args:
            api_key: Clé API Mistral (ou depuis MISTRAL_API_KEY env var)
            model: Nom du modèle à utiliser
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Clé API Mistral manquante. Définissez MISTRAL_API_KEY "
                "dans votre environnement ou passez api_key en paramètre."
            )

        self.model = model
        self.client = MistralSDKClient(api_key=self.api_key)
        logger.info(f"Client Mistral initialisé (SDK {MISTRAL_SDK_VERSION}, modèle: {model})")

    def _build_messages(self, prompt: str, system_prompt: Optional[str] = None) -> list:
        """Construit la liste des messages pour l'API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """
        Génère une réponse complète (non-streaming).

        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système optionnel
            temperature: Température de génération (0.0 - 1.0)
            max_tokens: Nombre maximum de tokens

        Returns:
            Texte de la réponse générée
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            if USE_LEGACY_API:
                # API v0.x : client.chat(messages=..., model=..., ...)
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # API v1.x+ : client.chat.complete(messages=..., model=..., ...)
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            # Extraction du contenu (structure similaire dans les deux versions)
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Erreur génération Mistral: {e}")
            raise

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Iterator[str]:
        """
        Génère une réponse en streaming (token par token).

        Args:
            prompt: Prompt utilisateur
            system_prompt: Prompt système optionnel
            temperature: Température de génération
            max_tokens: Nombre maximum de tokens

        Yields:
            Morceaux de texte générés
        """
        messages = self._build_messages(prompt, system_prompt)

        try:
            if USE_LEGACY_API:
                # API v0.x : client.chat_stream(...)
                stream = self.client.chat_stream(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                # API v1.x+ : client.chat.stream(...)
                stream = self.client.chat.stream(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Erreur streaming Mistral: {e}")
            raise

    def get_model_info(self) -> dict:
        """Retourne les informations sur le modèle utilisé."""
        return {
            "model": self.model,
            "sdk_version": MISTRAL_SDK_VERSION,
            "api_type": "legacy (v0.x)" if USE_LEGACY_API else "moderne (v1.x+)"
        }