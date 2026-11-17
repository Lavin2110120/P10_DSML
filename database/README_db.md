# SportSee — schéma PostgreSQL et ingestion (étape 2)

## Contenu et niveau de données

- `teams` : référentiel de 30 équipes (`team_code` comme clé primaire).
- `players` : identité locale par nom source normalisé (`player_id` technique). Ce nom n’est qu’une clé de dédoublonnage pour l’instant, pas un identifiant NBA pérenne.
- `stats` : mesures attachées au joueur, à l’équipe et à une clé de jeu de données. Elles sont des agrégats tels que fournis dans `Données NBA`, non des lignes de matchs. `season_label` est nullable et reste `NULL` à l’ingestion : aucune saison n’est inscrite dans le classeur, il ne faut donc pas la deviner. `dataset_key` rend l’upsert reproductible et distingue les imports/snapshots. `W + L = GP` est vérifié dans le pipeline et par contrainte en base. Les champs sont renommés explicitement (dont l’en-tête Excel lu comme heure `15:00:00` → `metric_15_00`). Les valeurs de pourcentage restent sur l’échelle source (le plus souvent 0–100); unités et sémantique ambiguës ne sont pas recalculées.
- `matches` : structure facultative pour de futurs matchs réellement datés (date, équipes, score). Elle ne représente pas les cumuls de `GP/W/L`, n’est pas alimentée par ce fichier et doit rester vide.
- `reports` : un enregistrement par publication initiale Reddit; titre, auteur, corps extrait et texte OCR complet conservé. Les commentaires, annonces et éléments d’interface ne sont pas transformés en rapports/commentaires structurés.

`schema.sql` est le DDL PostgreSQL de référence, avec clés primaires/étrangères, contrôles et index (dont recherche plein texte anglaise sur les publications). `load_excel_to_db.py` crée aussi les tables nécessaires lorsqu’il démarre, puis réalise des insertions idempotentes. Chaque ligne source passe par des modèles Pydantic distincts `PlayerRow` et `StatsRow`. Les lignes refusées sont détaillées dans `rejected_rows.csv` (ou `--rejects chemin.csv`). Le classeur attendu est la source fournie, feuille `Données NBA` avec en-tête sur la deuxième ligne physique, plus la feuille `Equipe` pour les libellés d’équipe.

## Dépendances et lancement

Python 3.10+ recommandé. Installer `requirements_db.txt`, puis définir `DATABASE_URL` avec un utilisateur/mot de passe fourni hors du code :

```sh
python -m pip install -r requirements_db.txt
export DATABASE_URL='postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME'
python load_excel_to_db.py
```

Par défaut, le script lit `/mnt/user-data/uploads/regular NBA.xlsx` et les quatre fichiers OCR `/mnt/user-data/uploads/Reddit 1.pdf.ocr.md` à `Reddit 4.pdf.ocr.md`. Options : `--xlsx chemin.xlsx`, `--database-url URL`, `--skip-reports`, `--rejects chemin.csv`. L’URL peut être fournie uniquement par `DATABASE_URL`. Pour éviter de divulguer les secrets, ne pas mettre de mot de passe dans un fichier livré ou une ligne de commande conservée dans l’historique.

Test de développement sans serveur PostgreSQL :

```sh
DATABASE_URL='sqlite:////tmp/sportsee_test.db' python load_excel_to_db.py
```

SQLite valide le flux d’import et les requêtes de contrôle, mais ne valide pas la syntaxe du DDL PostgreSQL (`schema.sql`) ni la recherche plein texte GIN. En PostgreSQL, exécuter également `psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f schema.sql` si l’on souhaite appliquer d’abord le DDL officiel. Les upserts PostgreSQL utilisent les contraintes `UNIQUE` déclarées dans ce schéma.

## Exemples de requêtes PostgreSQL

```sql
-- Effectif et moyenne de points bruts par équipe pour le snapshot courant.
SELECT t.team_code, t.team_name, count(*) AS joueurs,
       round(avg(s.points), 2) AS points_moyens_source
FROM stats s JOIN teams t USING (team_code)
WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1'
GROUP BY t.team_code, t.team_name
ORDER BY points_moyens_source DESC;

-- Top 10 des joueurs en points source (ne pas interpréter comme points par match).
SELECT p.source_name, s.team_code, s.points, s.gp
FROM stats s JOIN players p USING (player_id)
WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1'
ORDER BY s.points DESC LIMIT 10;

-- Comparer les moyennes de pourcentage de tir par équipe, à l’échelle Excel (0–100).
SELECT team_code, round(avg(field_goal_pct), 2) AS fg_pct_moyen,
       round(avg(true_shooting_pct), 2) AS ts_pct_moyen
FROM stats
WHERE dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1'
GROUP BY team_code ORDER BY fg_pct_moyen DESC;

-- Matches réels disponibles : attendu zéro pour les sources présentes.
SELECT count(*) AS matches_reels FROM matches;
```

## Rapports Reddit et limites

Les quatre URL/IDs et les titres/auteurs visibles sont enregistrés selon le rapport d’inspection. `body_text` retient le texte initial détecté avant URL, compteurs/commentaires, publicité ou interface; `raw_ocr_text` permet de revenir à toute la transcription. L’OCR de Reddit 3, page 2, contient une rangée Reggie Miller mal alignée (`2012 | 111`), conservée uniquement dans le brut et non convertie en données statistiques. Reddit 4, page 10, a un texte d’image illisible signalé dans l’OCR; pas de nouvel OCR lancé. Ces éléments ne sont pas réparés ni interprétés. Les commentaires Reddit n’ont pas d’identifiants fiables extraits et ne sont pas structurés.

Les données Excel comportent 569 joueurs, 45 champs par ligne et aucun champ saison/date/match. Les contraintes vérifient les contrôles observables, mais pas les définitions métier contradictoires du dictionnaire. Les agrégations affichées ci-dessus calculent la moyenne des valeurs sources, sans inventer d’unité ni attribuer de saison.
