from __future__ import annotations

# Test-only SDK doubles; project imports rely on the editable installation.
import importlib.util
import os
import pickle
import sys
import types
from pathlib import Path
import numpy as np

class FakeIndexFlatL2:
    def __init__(self, dimension):
        self.d = dimension
        self._vectors = np.empty((0, dimension), dtype=np.float32)
    @property
    def ntotal(self): return len(self._vectors)
    def add(self, vectors): self._vectors = np.concatenate((self._vectors, np.asarray(vectors, dtype=np.float32)))
    def search(self, queries, k):
        distances = ((np.asarray(queries)[:, None, :] - self._vectors[None, :, :]) ** 2).sum(axis=2)
        ids = np.argsort(distances, axis=1)[:, :k]
        return np.take_along_axis(distances, ids, axis=1), ids.astype(np.int64)
def _write_index(index, path):
    with open(path, "wb") as stream: pickle.dump((index.d, index._vectors), stream)
def _read_index(path):
    with open(path, "rb") as stream: dim, vectors = pickle.load(stream)
    index = FakeIndexFlatL2(dim); index.add(vectors); return index

class _StubMeta(type):
    """Metaclass making arbitrary class-level SDK attributes callable."""
    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        def _call(*args, **kwargs):
            return _make_stub_class(f"{cls.__name__}.{name}")()
        return _call


class _StubObject(metaclass=_StubMeta):
    """Harmless stand-in instance supporting dynamic SDK-shaped use."""
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def from_pandas(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    @classmethod
    def from_list(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        def _call(*args, **kwargs):
            return _make_stub_class(name)()
        return _call

    def __call__(self, *args, **kwargs):
        return _make_stub_class(type(self).__name__)()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(())


_STUB_CLASSES = {}
def _make_stub_class(name):
    """Return a cached permissive class accepting arbitrary construction."""
    if name not in _STUB_CLASSES:
        _STUB_CLASSES[name] = _StubMeta(name.rsplit('.', 1)[-1], (_StubObject,), {"__module__": __name__})
    return _STUB_CLASSES[name]


class _StubModule(types.ModuleType):
    """Optional dependency module that lazily caches arbitrary stub classes."""
    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _make_stub_class(name)
        setattr(self, name, value)
        return value


def _find_real_module(name):
    """Indique si un module peut etre importe sans charger le paquet."""
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError, AttributeError):
        return None


def _install_stub(name, *, force=False, package=False):
    """Enregistre un stub force ou uniquement si le paquet est absent."""
    if not force and _find_real_module(name) is not None:
        return None
    module = _StubModule(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


def install_stubs():
    """Installe en force les SDK cibles; conserve le stubbing conditionnel."""
    global faiss, mistralai, datasets, ragas, metrics, llms, embeddings, chat

    # Ces SDK sont toujours factices pour rendre les imports independants des versions.
    mistralai = _install_stub("mistralai", force=True, package=True)
    chat = _install_stub("langchain_mistralai", force=True, package=True)
    ragas = _install_stub("ragas", force=True, package=True)
    metrics = _install_stub("ragas.metrics", force=True, package=True)
    llms = _install_stub("ragas.llms", force=True, package=True)
    embeddings = _install_stub("ragas.embeddings", force=True, package=True)
    ragas.metrics = metrics
    ragas.llms = llms
    ragas.embeddings = embeddings

    # Les autres dependances optionnelles restent reelles lorsqu'elles existent.
    faiss = _install_stub("faiss", package=True)
    if isinstance(faiss, _StubModule):
        faiss.IndexFlatL2 = FakeIndexFlatL2
        faiss.write_index = _write_index
        faiss.read_index = _read_index

    datasets = _install_stub("datasets", package=True)
    for name in (
        "langchain", "langchain_core", "langchain_community",
        "langchain_text_splitters", "langchain_huggingface", "httpx",
    ):
        _install_stub(name, package=True)


# Valeurs factices non sensibles; setdefault preserve la configuration existante.
for _name, _value in {
    "DISABLE_SSL_VERIFY": "0",
    "APP_TITLE": "NBA Analyst AI",
    "APP_NAME": "NBA",
    "MISTRAL_API_KEY": "test-key",
    "MISTRAL_MODEL": "mistral-small-latest",
    "MISTRAL_EMBEDDING_MODEL": "mistral-embed",
    "CHUNK_SIZE": "1500",
    "CHUNK_OVERLAP": "150",
    "EMBEDDING_BATCH_SIZE": "32",
    "SEARCH_K": "5",
}.items():
    os.environ.setdefault(_name, _value)


install_stubs()

# A callable SDK facade with realistic nested endpoints for offline imports.
class _FakeChat:
    def complete(self, **kwargs):
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=""))])
    def stream(self, **kwargs):
        return iter(())
class _FakeEmbeddings:
    def create(self, **kwargs):
        items = kwargs.get("inputs", [])
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.0, 0.0]) for _ in items])
class _FakeMistralClient:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.chat = _FakeChat()
        self.embeddings = _FakeEmbeddings()
setattr(mistralai, "Mistral", _FakeMistralClient)
sys.modules.setdefault("rag_test_support", sys.modules[__name__])

import pytest

# Fixture paths are relative to the project root; imports come from the editable install.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================
# Fixtures reutilisables
# ============================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Chemin absolu vers la racine du projet."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_raw_dir(project_root: Path) -> Path:
    """Dossier data/raw/ (documents sources)."""
    p = project_root / "data" / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture(scope="session")
def vector_db_dir(project_root: Path) -> Path:
    """Dossier vector_db/ (index FAISS)."""
    p = project_root / "vector_db"
    p.mkdir(parents=True, exist_ok=True)
    return p


@pytest.fixture
def sample_text() -> str:
    """Petit texte de test, deterministe."""
    return (
        "LeBron James a marque 38 points face aux Celtics lors du Game 6 "
        "des finales NBA 2024. Les Lakers se sont imposes 106-94."
    )


@pytest.fixture
def sample_chunks(sample_text: str):
    """Liste de chunks simules pour les tests unitaires."""
    return [
        {"id": "c1", "text": sample_text, "metadata": {"source": "test"}},
        {"id": "c2", "text": "Anthony Davis a ajoute 17 rebonds.", "metadata": {"source": "test"}},
    ]