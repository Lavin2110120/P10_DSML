# ============================================
# Makefile - P10 DSML Projet RAG Mistral
# ============================================

SHELL := pwsh.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command

# Détection automatique du venv (.venv ou venv)
VENV_DIR := $(if $(wildcard .venv/Scripts/python.exe),.venv,$(if $(wildcard venv/Scripts/python.exe),venv,.venv))

PYTHON := $(VENV_DIR)\Scripts\python.exe
PIP := python -m pip
PYTEST := python -m pytest
STREAMLIT := python -m streamlit

.PHONY: help install setup data vectorstore test evaluate app lint format clean all

help:
    @Write-Host "`n=== P10 DSML - Commandes disponibles ===" -ForegroundColor Cyan
    @Write-Host "  make setup       - Installation complète (dépendances + arborescence)" -ForegroundColor White
    @Write-Host "  make install     - Installer les dépendances Python uniquement" -ForegroundColor White
    @Write-Host "  make data        - Ingérer documents (data/raw → data/processed)" -ForegroundColor White
    @Write-Host "  make vectorstore - Créer l'index FAISS (vector_db)" -ForegroundColor White
    @Write-Host "  make test        - Lancer les tests pytest avec couverture" -ForegroundColor White
    @Write-Host "  make evaluate    - Évaluer le RAG avec RAGAS" -ForegroundColor White
    @Write-Host "  make app         - Lancer l'interface Streamlit" -ForegroundColor White
    @Write-Host "  make lint        - Vérifier le code (ruff)" -ForegroundColor White
    @Write-Host "  make format      - Formater le code (black)" -ForegroundColor White
    @Write-Host "  make clean       - Nettoyer fichiers générés" -ForegroundColor White
    @Write-Host "  make all         - Pipeline complet : setup → data → vectorstore → test`n" -ForegroundColor White

install:
    @Write-Host "`n[1/6] Installation des dépendances..." -ForegroundColor Yellow
    pip install --upgrade pip
    pip install -r requirements.txt

setup: install
    @Write-Host "`n[2/6] Création de l'arborescence..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Force -Path data\raw, data\processed, vector_db, database, logs | Out-Null
    @Write-Host "Dossiers créés : data/raw, data/processed, vector_db, database, logs" -ForegroundColor Green

data:
    @Write-Host "`n[3/6] Traitement des documents..." -ForegroundColor Yellow
    python -m src.ingestion.data_loader

vectorstore:
    @Write-Host "`n[4/6] Création du vector store FAISS..." -ForegroundColor Yellow
    python -m src.rag.vector_store

test:
    @Write-Host "`n[5/6] Exécution des tests..." -ForegroundColor Yellow
    pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

evaluate:
    @Write-Host "`nÉvaluation RAGAS..." -ForegroundColor Yellow
    python -m src.evaluation.evaluate_ragas

app:
    @Write-Host "`nLancement de Streamlit..." -ForegroundColor Yellow
    streamlit run app/MistralChat.py

lint:
    @Write-Host "`nVérification du code (ruff)..." -ForegroundColor Yellow
    python -m ruff check src/ tests/ app/ --fix

format:
    @Write-Host "`nFormatage du code (black)..." -ForegroundColor Yellow
    python -m black src/ tests/ app/ --line-length 100

clean:
    @Write-Host "`nNettoyage..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force data\processed\*, vector_db\*, .pytest_cache, htmlcov, .coverage -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    @Write-Host "Nettoyage terminé." -ForegroundColor Green

all: setup data vectorstore test
    @Write-Host "`n=== Pipeline complet terminé avec succès ! ===" -ForegroundColor Green