"""Offline test fixtures and optional-SDK stubs for the Gemini RAG project."""
from __future__ import annotations
import os, pickle, sys, types
from pathlib import Path
import numpy as np
import pytest

for key, value in {
    "DISABLE_SSL_VERIFY": "0", "GEMINI_API_KEY": "unit-test-key",
    "GOOGLE_API_KEY": "unit-test-key", "GEMINI_MODEL": "gemini-test",
    "GEMINI_EMBEDDING_MODEL": "text-embedding-001", "EMBEDDING_DIM": "3072",
}.items():
    os.environ.setdefault(key, value)

class FakeIndex:
    def __init__(self, dimension): self.d, self.vectors = dimension, np.empty((0, dimension), np.float32)
    @property
    def ntotal(self): return len(self.vectors)
    def add(self, vectors): self.vectors = np.concatenate([self.vectors, np.asarray(vectors, dtype=np.float32)])
    def search(self, queries, k):
        scores = np.asarray(queries) @ self.vectors.T
        ids = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, ids, axis=1), ids.astype(np.int64)

def _write_index(index, path):
    with open(path, "wb") as f: pickle.dump((index.d, index.vectors), f)
def _read_index(path):
    with open(path, "rb") as f: d, v = pickle.load(f)
    index=FakeIndex(d); index.add(v); return index
def _normalize(v):
    norms=np.linalg.norm(v, axis=1, keepdims=True); np.divide(v, norms, out=v, where=norms!=0)

faiss=types.ModuleType("faiss")
faiss.IndexFlatIP=FakeIndex; faiss.write_index=_write_index; faiss.read_index=_read_index; faiss.normalize_L2=_normalize
sys.modules["faiss"]=faiss

# The vector-store source imports this narrow Google SDK submodule at import time.
google=types.ModuleType("google"); google.__path__=[]
genai=types.ModuleType("google.genai"); genai.__path__=[]
errors=types.ModuleType("google.genai.errors")
class ClientError(Exception):
    def __init__(self, message="client error", code=None): super().__init__(message); self.code=code
errors.ClientError=ClientError; genai.errors=errors; google.genai=genai
sys.modules.setdefault("google", google); sys.modules["google.genai"]=genai; sys.modules["google.genai.errors"]=errors

# evaluate_ragas.py still imports its legacy Mistral name; this test-only shim
# keeps module collection offline while tests replace that client explicitly.
legacy=types.ModuleType("src.llm.mistral_client")
legacy.MistralClient=type("MistralClient", (), {})
sys.modules["src.llm.mistral_client"]=legacy
ragas=types.ModuleType("ragas"); ragas.evaluate=None; ragas.__path__=[]
sys.modules["ragas"]=ragas

@pytest.fixture
def offline_embeddings():
    def embed(texts, **kwargs):
        return [[float((sum(map(ord, t)) + i) % 13 + 1) for i in range(3072)] for t in texts]
    return embed

@pytest.fixture
def project_root(): return Path(__file__).resolve().parents[1]
