# ============================================
# tests/conftest.py
# Fixtures pytest partagees pour toute la suite.
# Ajoute la racine du projet et src/ aux chemins d'import
# pour permettre `from src.xxx import ...` et les imports internes
# tels que `from rag.xxx import ...`, quel que soit le cwd.
# ============================================
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# --- Resolution du PYTHONPATH ---------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

# La racine permet `from src.xxx import ...`, src/ permet les imports
# internes non prefixes (`from rag.xxx import ...`, `from llm.xxx import ...`).
for path in (PROJECT_ROOT, SRC_PATH):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Expose aussi les deux chemins via PYTHONPATH (processus enfants, Streamlit, etc.)
pythonpath_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
for path in (SRC_PATH, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in pythonpath_entries:
        pythonpath_entries.insert(0, path_str)
os.environ["PYTHONPATH"] = os.pathsep.join(e for e in pythonpath_entries if e)


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