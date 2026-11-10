# Makefile — P10 DSML Projet RAG Mistral
# Recipes use GNU make and Python commands (no PowerShell-specific syntax).
# On Windows, run with GNU Make and ensure python, pip-compile, and pytest are on PATH.

DISABLE_SSL_VERIFY ?= 0
export DISABLE_SSL_VERIFY

.PHONY: help install setup data vectorstore test evaluate app lint format lock clean all

help:
	@echo "=== P10 DSML - Commandes disponibles ==="
	@echo "  make setup       - Installation complète (dépendances + configuration du projet)"
	@echo "  make install     - Installer le projet et ses dépendances de développement"
	@echo "  make data        - Ingérer les documents puis construire l'index (data/raw → vector_db)"
	@echo "  make vectorstore - (Re)construire le vector store FAISS"
	@echo "  make test        - Lancer pytest avec couverture"
	@echo "  make evaluate    - Évaluer le RAG avec RAGAS"
	@echo "  make app         - Lancer l'interface Streamlit"
	@echo "  make lint        - Vérifier src, app et tests (ruff, black, mypy)"
	@echo "  make format      - Formater le code avec black"
	@echo "  make lock        - Générer requirements-lock.txt avec pip-compile"
	@echo "  make clean       - Nettoyer couverture et caches"
	@echo "  make all         - Pipeline complet : install → data → vectorstore → test"

install:
	python -m pip install -e ".[dev]"

# Retained for compatibility with existing workflows; editable install includes setup.
setup: install

data:
	python -m src.ingestion.data_loader
	python -m src.ingestion.indexer

vectorstore:
	python -m src.rag.vector_store

test:
	python -m pytest tests/ --cov=src --cov-report=term-missing

evaluate:
	python -m src.evaluation.evaluate_ragas

app:
	python -m streamlit run app/MistralChat.py

lint:
	python -m ruff check src app tests
	python -m black --check src app tests
	python -m mypy src app

format:
	python -m black src app tests --line-length 100

lock:
	pip-compile --extra dev pyproject.toml --output-file requirements-lock.txt

clean:
	python -c "import pathlib, shutil; roots=[pathlib.Path('.coverage'), pathlib.Path('htmlcov'), pathlib.Path('.pytest_cache'), pathlib.Path('.mypy_cache'), pathlib.Path('.ruff_cache')]; [shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True) for p in roots]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]"

all: install data vectorstore test
