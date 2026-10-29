# 🏀 SportSee - Assistant IA d'Analyse de Performance Sportive

Système RAG (Retrieval-Augmented Generation) pour aider les coachs à interroger
les archives textuelles et données de matchs de basketball.

## 🎯 Objectif
Permettre aux entraîneurs de poser des questions en langage naturel :
- "Quel joueur a le meilleur % à 3 points sur les 5 derniers matchs ?"
- "Compare les statistiques de rebonds domicile vs extérieur"

## 🏗️ Architecture

┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│   Streamlit │────▶│  Retriever   │────▶│  FAISS      │────▶│ Mistral  │
│   (UI)      │     │  (LangChain) │     │  (Vector DB)│     │  (LLM)   │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
        │                                              │
        └──────────────────────────────────────────────┘

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS pour trouver les documents pertinents
- 🤖 **Génération de réponses** avec les modèles Mistral (Small ou Large)
- ⚙️ **Paramètres personnalisables** (modèle, nombre de documents, score minimum)

## 📦 Installation

### Prérequis
- Python 3.9+
- Clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### Setup
```bash
# 1. Cloner
git clone "https://github.com/Lavin2110120/P10_DSML"
cd P10_DSML

# 2. Environnement
python -m venv venv
venv\Scripts\activate # Windows
# Linux: source venv/bin/activate  


# 3. Dépendances
pip install -r requirements.txt

# 4. Configuration
cp .env.example .env

## Structure du projet
P10_DSML/
├── 📁 src/                    # Code source organisé
│   ├── 📁 ingestion/          # Préparation données
│   │   ├── indexer.py         
│   │   └── data_loader.py     
│   ├── 📁 rag/                # Chaîne RAG
│   │   ├── vector_store.py    
│   │   └── retriever.py       # À créer si besoin
│   ├── 📁 llm/                # Interaction Mistral
│   │   └── mistral_client.py  # (extrait de MistralChat.py)
│   └── 📁 evaluation/         # Scripts d'audit
│       └── evaluate_ragas.py  
├── 📁 app/                    # Applications
│   └── MistralChat.py         
├── 📁 data/
│   ├── 📁 raw/                # Données brutes (Excel SportSee, PDF commentaires)
│   └── 📁 processed/          # Données nettoyées
├── 📁 vector_db/              # Index FAISS 
├── 📁 database/               # SQLite/PostgreSQL 
├── 📁 tests/                  # Tests unitaires
├── 📁 notebooks/              # Analyses exploratoires
├── .env
├── .gitignore
├── README.md                  
└── requirements.txt

🚀 Utilisation
1. Placer les données

    Données Excel (matchs, stats) → data/raw/
    Commentaires textuels (PDF/TXT) → inputs/

2. Indexer les documents

python src/ingestion/indexer.py

Génère l'index FAISS dans vector_db/
3. Lancer l'assistant

streamlit run app/MistralChat.py

📊 Évaluation

python src/evaluation/evaluate_ragas.py

Génère un rapport de métriques RAGAS (fidélité, pertinence).

👥 Équipe

    Data Scientist : Vincent
    Mentor : Sarah (R&D SportSee)

