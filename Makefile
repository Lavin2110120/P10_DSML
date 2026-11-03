# ============================================
# Makefile - P10 DSML Projet RAG Mistral
# ============================================

# Forcer PowerShell comme shell
SHELL := powershell.exe
.SHELLFLAGS := -NoProfile -ExecutionPolicy Bypass -Command


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
	@Write-Host "  make all         - Pipeline complet : install → data → vectorstore → test`n" -ForegroundColor White

install:
	@Write-Host "`n[1/4] Installation des dépendances..." -ForegroundColor Yellow
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt


data:
	@Write-Host "`n[2/4] Traitement des documents..." -ForegroundColor Yellow
	$$env:PYTHONPATH="src"; $$env:DISABLE_SSL_VERIFY="1"; $$env:PYTHONHTTPSVERIFY="0"; python -m ingestion.data_loader
	$$env:PYTHONPATH="src"; $$env:DISABLE_SSL_VERIFY="1"; $$env:PYTHONHTTPSVERIFY="0"; python -m ingestion.indexer

vectorstore:
	@Write-Host "`n[3/4] Création du vector store FAISS..." -ForegroundColor Yellow
	$$env:PYTHONPATH="src"; python -m rag.vector_store

test:
	@Write-Host "`n[4/4] Exécution des tests..." -ForegroundColor Yellow
	$$env:PYTHONPATH="src"; python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

evaluate:
	@Write-Host "`nÉvaluation RAGAS..." -ForegroundColor Yellow
	$$env:PYTHONPATH="src"; python -m evaluation.evaluate_ragas

app:
	@Write-Host "`nLancement de Streamlit..." -ForegroundColor Yellow
	python -m streamlit run app/MistralChat.py

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

all: install data vectorstore test
	@Write-Host "`n=== Pipeline complet terminé avec succès ! ===" -ForegroundColor Green