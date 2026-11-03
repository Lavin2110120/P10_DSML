from __future__ import annotations
from typing import Any

import httpx
from rag import config
from rag.ssl_bypass import disable_ssl_verification

# Apply before creating SDK clients so all downstream httpx users inherit it.
_SSL_BYPASS_ENABLED = disable_ssl_verification()
try:
    from mistralai import Mistral
except Exception:  # pragma: no cover - optional dependency
    Mistral = None


class MistralClient:
    def __init__(self, api_key=None, model=None, embedding_model=None, **kwargs):
        self.api_key = api_key if api_key is not None else config.MISTRAL_API_KEY
        if not self.api_key:
            raise ValueError("clé API Mistral manquante")
        self.model = model if model is not None else config.MODEL_NAME
        if not self.model:
            raise ValueError("MODEL_NAME ne peut pas être vide")
        self.embedding_model = (embedding_model if embedding_model is not None
                                else config.EMBEDDING_MODEL)
        if Mistral is None:
            raise RuntimeError("SDK Mistral indisponible")
        client_kwargs = {}
        if _SSL_BYPASS_ENABLED:
            client_kwargs.update(client=httpx.Client(verify=False),
                                 async_client=httpx.AsyncClient(verify=False))
        self.client = Mistral(api_key=self.api_key, **client_kwargs)

    @staticmethod
    def _messages(prompt, system_prompt=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def generate(self, prompt, system_prompt=None, temperature=0.7, max_tokens=500,
                 top_p=1.0, **kwargs):
        try:
            resp = self.client.chat.complete(
                model=self.model, messages=self._messages(prompt, system_prompt),
                temperature=temperature, max_tokens=max_tokens, top_p=top_p)
            choices = getattr(resp, "choices", []) or []
            return ((getattr(getattr(choices[0], "message", None), "content", None) or "")
                    if choices else "")
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def generate_response(self, prompt, system_prompt=None, temperature=0.7,
                          max_tokens=500, **kwargs):
        return self.generate(prompt, system_prompt, temperature, max_tokens, **kwargs)

    def generate_stream(self, prompt, system_prompt=None, temperature=0.7,
                        max_tokens=500, **kwargs):
        try:
            stream = self.client.chat.stream(
                model=self.model, messages=self._messages(prompt, system_prompt),
                temperature=temperature, max_tokens=max_tokens)
            for chunk in stream:
                data = getattr(chunk, "data", None)
                choices = getattr(data, "choices", []) if data is not None else []
                if choices:
                    content = getattr(getattr(choices[0], "delta", None), "content", None)
                    if content:
                        yield content
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def embeddings(self, texts):
        if not texts:
            return []
        active_model = (config.EMBEDDING_MODEL if self.embedding_model == "mistral-embed"
                        else self.embedding_model)
        if not active_model:
            raise ValueError("EMBEDDING_MODEL non configuré")
        try:
            res = self.client.embeddings.create(model=active_model, inputs=list(texts))
            return [item.embedding for item in res.data]
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def generate_embeddings(self, texts, batch_size=32):
        return self.embeddings(texts)

    def chat_with_context(self, question, context_chunks, **kwargs):
        context = "\n\n".join(context_chunks)
        prompt = f"Contexte:\n{context}\nQuestion: {question}"
        return {"answer": self.generate(prompt, max_tokens=kwargs.get("max_tokens", 500))}
