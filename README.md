# Assistant RAG Basket — P10 DSML

Application de recherche et de question-réponse sur des documents de basket/NBA. Le projet utilise Gemini pour la génération et les embeddings, FAISS pour la recherche locale, Pydantic AI pour les réponses validées, Logfire pour une observabilité facultative et RAGAS pour l’évaluation.

## Prérequis

- Python 3.10 ou ultérieur.
- Une clé Gemini (`GOOGLE_API_KEY` ou `GEMINI_API_KEY`), configurée localement dans `.env`.
- GNU Make pour les commandes `make`.
- Pour indexer : documents placés dans `data/raw/`. Pour l’évaluation métier : `data/evaluation/questions_metier.csv`.
- Pour le texte des PDF scannés : Tesseract OCR doit être installé comme programme système. Le paquet Python `pytesseract` et Pillow sont également utilisés; l’OCR peut être ignoré si ces composants ne sont pas disponibles.

Ne jamais ajouter de clés ou de jetons réels à ce README, au dépôt ou aux rapports partagés. `.env` est un fichier local.

## Installation

Depuis la racine du dépôt :

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env       # Windows : Copy-Item .env.example .env
```

Renseigner ensuite localement `GOOGLE_API_KEY` ou `GEMINI_API_KEY`. `LOGFIRE_TOKEN` est facultatif. Vérifier les réglages de TLS dans `.env`; ne pas désactiver les contrôles de certificats en usage normal.

## Arborescence

L’arborescence ci-dessous suit le listing du dépôt fourni avec les fichiers :

```text
P10_DSML/
├── app/
│   └── GeminiChat.py
├── data/
│   ├── evaluation/
│   │   ├── questions_metier.csv
│   │   ├── ragas_scores.csv            # généré
│   │   └── ...                         # caches et autres rapports générés
│   ├── processed/                      # documents.json généré
│   └── raw/                            # PDF, XLSX et TXT sources
├── database/
├── logs/
├── reports/                            # exports de compatibilité générés
├── src/
│   ├── evaluation/
│   │   └── evaluate_ragas.py
│   ├── ingestion/
│   │   ├── data_loader.py
│   │   └── indexer.py
│   ├── llm/
│   │   ├── gemini_client.py
│   │   └── structured_rag.py
│   ├── rag/
│   │   └── vector_store.py
│   ├── config.py
│   ├── models.py
│   ├── observability.py
│   └── ssl_bypass.py
├── tests/
├── vector_db/                          # index FAISS et métadonnées générés
├── .env.example
├── Makefile
├── pyproject.toml
└── README.md
```

Les répertoires `src/` et `app/` sont des chemins valides dans l’arborescence fournie. L’application est `app/GeminiChat.py` (et non un fichier à la racine). Les données produites localement, les caches et les rapports n’ont pas besoin d’être committés.

## Utilisation

### Ingestion et indexation

Déposer les fichiers source `.txt`, `.pdf` ou `.xlsx` sous `data/raw/`, puis :

```bash
python -m src.ingestion.data_loader
python -m src.ingestion.indexer
```

Ou lancer `make data` depuis la racine. Le chargeur extrait le texte et écrit le corpus traité sous `data/processed/`; l’indexeur calcule les embeddings Gemini et construit l’index FAISS dans `vector_db/`. `make vectorstore` permet de reconstruire le magasin à partir des données préparées.

### Application Streamlit

```bash
streamlit run app/GeminiChat.py
# ou
make app
```

La recette `app` du Makefile fourni pointe déjà vers `app/GeminiChat.py`.

### Évaluation RAGAS

Placer les questions dans `data/evaluation/questions_metier.csv`, puis :

```bash
python -m src.evaluation.evaluate_ragas
python -m src.evaluation.evaluate_ragas --limit 5
python -m src.evaluation.evaluate_ragas --categories simple complexe --limit 10
# ou
make evaluate
```

Le CSV des questions doit inclure `question`, `ground_truth_expected` et `category`. Les catégories actuellement validées dans le code sont `simple`, `complexe`, `bruitée` et `hors-périmètre/numérique`. Les appels Gemini et RAGAS nécessitent un accès réseau et peuvent consommer du quota.

## Mode de réponse : contrôle et limites

**Écart entre les fichiers Python joints :** l’arborescence jointe indique les chemins `src/evaluation/evaluate_ragas.py` et `src/llm/structured_rag.py`, mais le fichier aplati nommé `evaluate_ragas.py` fourni ne contient pas de mode structuré : il ne référence ni `generate_structured_answer`, ni `mode`, et ne crée pas de span Logfire par question. Les copies jointes `evaluate_ragas(3).py` et `evaluate_ragas(4).py` contiennent une implémentation structurée plus récente. Les explications ci-dessous décrivent ce comportement récent; elles ne doivent pas être attribuées au fichier non suffixé sans synchroniser celui-ci avec cette version.

Dans cette implémentation structurée plus récente, la colonne de détail qui enregistre le mode est exactement **`mode`** dans `data/evaluation/ragas_detail_par_question.csv`; les valeurs admises sont **`structured`** et **`fallback`**. Le champ `mode` est également préservé dans les caches (`data/evaluation/cache_reponses.json` et `data/evaluation/answers_cache.jsonl`) et dans `reports/ragas_input.csv` (ligne de résultat). `ragas_scores.csv` contient les scores RAGAS, pas le détail de mode. Les caches anciens dépourvus d’un mode reconnu sont régénérés plutôt que classés arbitrairement.

Dans la classe `RAGSystem.query` de cette version, `mode` est initialisé à `structured`. Si l’appel à `generate_structured_answer(...)` ou la validation `RAGAnswer` lève une exception, le code d’évaluation fait un appel direct à Gemini et marque `mode="fallback"`. **Faux `structured` possible :** `src/llm/structured_rag.py` attrape elle-même les exceptions Pydantic AI, journalise un repli Gemini et renvoie ensuite un objet `RAGAnswer`. L’évaluateur ne voit alors aucune exception et conserve `mode="structured"`, même si le texte a été généré par Gemini en fallback interne. Une ligne `structured` ne prouve donc pas à elle seule que Pydantic AI a généré la réponse. Chercher le message INFO exact **`Pydantic AI indisponible ou génération structurée échouée; repli Gemini (...)`** dans les logs Python; activer INFO pour l’observer. Un `fallback` dans la colonne CSV ne signale que le repli explicite de `RAGSystem.query` après une exception qui a échappé à `generate_structured_answer` ou après échec de validation.

Dans ces copies récentes, Logfire crée une span **`ragas.question`** avec `category` et `question_id`, mais n’y ajoute pas `mode`. La span Logfire seule ne distingue donc pas les modes; le message ci-dessus est un log Python (`src.llm.structured_rag`), non un événement Logfire. Dans le fichier non suffixé `evaluate_ragas.py` joint, il n’y a pas de span `ragas.question` non plus.
## Rapports, tests et maintenance

Les sorties d’évaluation détaillées sont écrites sous `data/evaluation/` (`ragas_detail_par_question.csv`, `ragas_scores.csv` et rapports par catégorie). Des exports de compatibilité sont écrits sous `reports/`. Lancer les tests et contrôles avec :

```bash
make test
make lint
```

Pour consulter les commandes disponibles : `make help`. Pour verrouiller les dépendances, la cible `make lock` utilise `pip-compile` (pip-tools doit être installé). Après un changement de modèle ou de dimension d’embedding, vérifier la cohérence avec l’index existant et le reconstruire si nécessaire.
