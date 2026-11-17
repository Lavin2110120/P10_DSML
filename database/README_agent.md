# SportSee — couche agent SQL (étape 2, suite)

## Composants et branchement

- `sql_tool.py` construit le tool LangChain `sportsee_sql_readonly` à partir du schéma défini dans `schema.sql`. Il utilise huit exemples few-shot en français, SQLAlchemy, une validation SELECT-only, une limite de lignes, un timeout et une boucle de correction (une tentative par défaut, deux au maximum).
- `agent_sql_integration.py` est un module autonome de démonstration LangGraph ReAct. Le prototype d'agent d'origine n'étant pas disponible, il s'intègre ainsi : ajoutez `make_sql_tool(...)` à la liste `tools` de l'agent existant et utilisez `SYSTEM_PROMPT` dans son prompt système. Passez le retriever Reddit déjà créé à `build_agent(retriever=...)`; sans retriever, seul l'outil SQL est branché.
- Les clés API ne sont jamais codées dans le module. Le modèle peut être fourni directement à `make_sql_tool(llm=...)` / `build_agent(model=...)`, ou créé via la configuration d'environnement ci-dessous.

## Installation

Python 3.10+ recommandé. Depuis le dossier du projet :

```sh
python -m pip install -r requirements_db.txt
python -m pip install -r requirements_agent.txt
```

Le choix des extras modèle n'est pas implicite : `langchain-openai` et `langchain-mistralai` figurent dans le fichier; configurez l'un des fournisseurs autorisés. `DATABASE_URL` est une URL SQLAlchemy vers PostgreSQL en production.

```sh
export DATABASE_URL='postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME'
export SQL_LLM_PROVIDER='openai'       # ou mistral
export SQL_LLM_MODEL='gpt-4o-mini'     # facultatif; valeur par défaut OpenAI
export OPENAI_API_KEY='...'            # pour OpenAI; sinon MISTRAL_API_KEY
# export MISTRAL_API_KEY='...'
export SQL_MAX_ROWS='100'              # facultatif
export SQL_TIMEOUT_SECONDS='8'         # facultatif
python agent_sql_integration.py
```

Ne commitez pas de secrets ou d'URL avec mot de passe. Pour Mistral, un modèle par défaut `mistral-small-latest` est choisi si `SQL_LLM_MODEL` est absent. Le langage modèle est injecté en paramètre pour les tests et les déploiements spécifiques.

## Développement / test sans PostgreSQL ni LLM

Les tests utilisent une petite base SQLite éphémère et un faux LLM. Ils ne contactent ni API, ni serveur PostgreSQL. `SQLToolRunner` accepte un modèle factice, un `Engine` SQLAlchemy ou une URL SQLite. Le classeur donné a également été chargé localement avec le script d'ingestion existant : 569 lignes statistiques acceptées, 569 joueurs, 30 équipes et 0 match; cette base reste un artefact temporaire du sandbox, pas un livrable.

```sh
python -m pytest -q test_sql_tool.py
```

Pour essayer l'agent en production (après installation des dépendances), fournissez la base et les variables modèle, puis :

```sh
python agent_sql_integration.py
```

Pour brancher un LLM LangChain quelconque :

```python
from sql_tool import make_sql_tool
from agent_sql_integration import SYSTEM_PROMPT
sql_tool = make_sql_tool(llm=mon_chat_model)
# Puis ajouter sql_tool aux outils de create_react_agent / agent existant.
```

## Exemples de questions

- « Quels sont les cinq joueurs avec le plus de points dans le jeu de données ? »
- « Quel joueur a le meilleur pourcentage à trois points avec au moins 20 matchs joués ? »
- « Compare les rebonds moyens de Boston et New York. »
- « Quelles équipes combinent le mieux points moyens et rebonds moyens ? »
- « Compare les passes décisives des joueurs ayant joué au moins 30 matchs. »

Les demandes chiffrées/statistiques relèvent de SQL. Les discussions, opinions et anecdotes issues des publications sont à chercher dans le retriever textuel Reddit, et non dans SQL. Pour les demandes mixtes, appeler les deux outils et distinguer les sources.

## Limites et garde-fous

Les tables et colonnes sont issues du vrai schéma SportSee (teams, players, stats, matches, reports). `stats` représente des agrégats/snapshot joueur-équipe; `gp` ne fournit pas les logs de matchs. Aucune saison n'est renseignée et `matches` est vide. Le tool bloque explicitement les requêtes formulées sur les derniers matchs, domicile/extérieur et la saison; il ne peut donc pas répondre à ces demandes. Les pourcentages gardent les valeurs sources et souvent leur échelle 0–100; `points` ne doit pas être présenté automatiquement comme points par match. Les requêtes agrégées ne constituent pas des statistiques NBA recalculées à partir de matchs.

Le validateur limite les statements à SELECT/WITH-SELECT, interdit les mots-clés d'écriture/admin/commentaires et multi-statements, plafonne LIMIT à `SQL_MAX_ROWS`, utilise une transaction read-only et le statement timeout PostgreSQL. SQLite active `PRAGMA query_only` et un progress handler. Une base déployée doit aussi utiliser un utilisateur PostgreSQL ayant uniquement des droits de lecture : la validation et la transaction sont des défenses en profondeur, pas un substitut aux permissions minimales.

## Résultats de validation locale

`python -m pytest -q test_sql_tool.py` : 6 tests passés. Un audit automatique vérifie que chaque identifiant table/colonne des huit exemples SQL existe dans `schema.sql`. Tests réalisés sur SQLite/faux LLM uniquement. Aucun appel à un fournisseur LLM, aucun essai de connexion PostgreSQL, ni vérification d'exécution des exemples en PostgreSQL n'a été fait dans ce sandbox.
