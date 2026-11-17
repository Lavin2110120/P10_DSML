"""Tool SQL en lecture seule pour SportSee (schéma de schema.sql)."""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Callable

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_core.tools import StructuredTool

MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "100"))
QUERY_TIMEOUT_SECONDS = int(os.getenv("SQL_TIMEOUT_SECONDS", "8"))

SCHEMA_DESCRIPTION = """
Schéma PostgreSQL SportSee (n'utiliser que ces tables/colonnes):
- teams(team_code CHAR(3), team_name TEXT): référentiel des équipes NBA.
- players(player_id BIGINT, source_name TEXT, created_at TIMESTAMPTZ): identité locale d'un joueur.
- stats(stats_id BIGINT, player_id BIGINT, team_code CHAR(3), season_label TEXT nullable,
  dataset_key TEXT, age SMALLINT, gp SMALLINT, wins SMALLINT, losses SMALLINT,
  minutes NUMERIC, points INTEGER, field_goals_made INTEGER, field_goals_attempted INTEGER,
  field_goal_pct NUMERIC, metric_15_00 INTEGER, three_pointers_attempted INTEGER,
  three_point_pct NUMERIC, free_throws_made INTEGER, free_throws_attempted INTEGER,
  free_throw_pct NUMERIC, offensive_rebounds INTEGER, defensive_rebounds INTEGER,
  rebounds INTEGER, assists INTEGER, turnovers INTEGER, steals INTEGER, blocks INTEGER,
  personal_fouls INTEGER, fantasy_points INTEGER, double_doubles INTEGER, triple_doubles INTEGER,
  plus_minus NUMERIC, off_rating NUMERIC, def_rating NUMERIC, net_rating NUMERIC,
  assist_pct NUMERIC, assist_to_turnover NUMERIC, assist_ratio NUMERIC,
  offensive_rebound_pct NUMERIC, defensive_rebound_pct NUMERIC, rebound_pct NUMERIC,
  turnover_ratio NUMERIC, effective_fg_pct NUMERIC, true_shooting_pct NUMERIC,
  usage_pct NUMERIC, pace NUMERIC, pie NUMERIC, possessions INTEGER,
  source_file TEXT, source_sheet TEXT, source_row INTEGER, loaded_at TIMESTAMPTZ).
  Une ligne = agrégat/snapshot joueur-équipe. gp est le nombre de matchs joués, pas une table match.
  Les pourcentages suivent l'échelle du classeur, souvent 0–100. points ne signifie pas
  nécessairement points par match. season_label est NULL pour ce jeu de données.
- matches(match_id, match_date, season_label, home_team_code, away_team_code, home_score,
  away_score, source_ref): structure future, vide avec les données actuellement fournies.
- reports(report_id, source_platform, subreddit, post_id, source_url, author, title, body_text,
  raw_ocr_text, capture_date_text, source_file, created_at): contenu textuel Reddit/OCR.
Relations : stats.player_id=players.player_id; stats.team_code=teams.team_code.
Filtrer les agrégats du classeur via dataset_key='regular-nba-donnees-nba-player-team-snapshot-v1'.
Aucune donnée par match, aucune saison documentée : impossible de répondre à « 5 derniers matchs »,
à domicile/extérieur ou à une saison précise. Ne jamais déduire/inventer ces informations.
""".strip()

FEW_SHOT_EXAMPLES = """
Exemples PostgreSQL — conserver les jointures et filtres de dataset lorsque pertinents.
Q: Quel joueur a le meilleur pourcentage à trois points, avec au moins 20 matchs joués ?
SQL: SELECT p.source_name, s.team_code, s.three_point_pct, s.gp FROM stats s JOIN players p USING (player_id) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' AND s.gp >= 20 ORDER BY s.three_point_pct DESC, p.source_name LIMIT 1

Q: Donne le top 5 des joueurs en points.
SQL: SELECT p.source_name, s.team_code, s.points, s.gp FROM stats s JOIN players p USING (player_id) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' ORDER BY s.points DESC, p.source_name LIMIT 5

Q: Compare les rebonds des joueurs de Boston et de New York.
SQL: SELECT t.team_name, ROUND(AVG(s.rebounds), 2) AS rebonds_moyens, MAX(s.rebounds) AS maximum FROM stats s JOIN teams t USING (team_code) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' AND t.team_name ILIKE ANY (ARRAY['%Boston%', '%New York%']) GROUP BY t.team_name ORDER BY rebonds_moyens DESC LIMIT 10

Q: Quelles équipes combinent le mieux moyenne de points et de rebonds ?
SQL: SELECT t.team_name, ROUND(AVG(s.points), 2) AS points_moyens, ROUND(AVG(s.rebounds), 2) AS rebonds_moyens, COUNT(*) AS joueurs FROM stats s JOIN teams t USING (team_code) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' GROUP BY t.team_code, t.team_name ORDER BY points_moyens DESC, rebonds_moyens DESC LIMIT 10

Q: Classe les joueurs ayant joué au moins 30 matchs selon leurs passes décisives.
SQL: SELECT p.source_name, s.team_code, s.assists, s.gp FROM stats s JOIN players p USING (player_id) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' AND s.gp >= 30 ORDER BY s.assists DESC LIMIT 10

Q: Quelles équipes ont le meilleur pourcentage moyen à 3 points parmi les joueurs à 10 matchs minimum ?
SQL: SELECT t.team_name, ROUND(AVG(s.three_point_pct), 2) AS pourcentage_moyen, COUNT(*) AS joueurs FROM stats s JOIN teams t USING (team_code) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' AND s.gp >= 10 GROUP BY t.team_code, t.team_name ORDER BY pourcentage_moyen DESC LIMIT 10

Q: Quel est le meilleur ratio passes décisives/pertes de balle ?
SQL: SELECT p.source_name, s.assist_to_turnover, s.gp FROM stats s JOIN players p USING (player_id) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' ORDER BY s.assist_to_turnover DESC LIMIT 10

Q: Compare points, victoires et taux d'usage moyen par équipe pour les joueurs à au moins 15 matchs.
SQL: SELECT t.team_name, ROUND(AVG(s.points), 2) AS points_moyens, SUM(s.wins) AS victoires, ROUND(AVG(s.usage_pct), 2) AS usage_moyen FROM stats s JOIN teams t USING (team_code) WHERE s.dataset_key = 'regular-nba-donnees-nba-player-team-snapshot-v1' AND s.gp >= 15 GROUP BY t.team_code, t.team_name ORDER BY points_moyens DESC LIMIT 10
""".strip()

OUT_OF_SCOPE_MESSAGE = ("Je ne peux pas répondre précisément : la base ne contient pas de statistiques "
                        "par match ni de saison documentée. Les demandes sur les 5 derniers matchs, "
                        "domicile/extérieur ou une saison précise sont hors périmètre. Aucune valeur "
                        "ne sera inventée.")


def _out_of_scope(question: str) -> bool:
    q = question.lower()
    patterns = (r"dernier(?:s|es)?\s+(?:\d+\s+)?match", r"\d+\s+dernier(?:s|es)?\s+match",
                r"domicile|extérieur|exterieur|home[- ]away", r"saison")
    return any(re.search(p, q) for p in patterns)


def validate_sql(sql: str, max_rows: int = MAX_ROWS) -> str:
    """Valide une requête de lecture et y impose une limite stricte de lignes."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("Requête SQL vide.")
    candidate = sql.strip()
    candidate = re.sub(r"^```(?:sql)?\s*|\s*```$", "", candidate, flags=re.I).strip()
    if "--" in candidate or "/*" in candidate or "*/" in candidate:
        raise ValueError("Commentaires SQL interdits.")
    if candidate.endswith(";"):
        candidate = candidate[:-1].rstrip()
    if ";" in candidate:
        raise ValueError("Un seul statement SQL est autorisé.")
    # La grammaire admise est SELECT (WITH ... SELECT compris); les mots de mutation sont interdits partout.
    if not re.match(r"^(SELECT|WITH)\b", candidate, re.I):
        raise ValueError("Seules les requêtes SELECT sont autorisées.")
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|MERGE|CALL|COPY|VACUUM|ATTACH|DETACH|PRAGMA|INTO|FOR\s+UPDATE|LOCK)\b", candidate, re.I):
        raise ValueError("Mot-clé SQL interdit dans la requête.")
    if not isinstance(max_rows, int) or max_rows < 1:
        raise ValueError("La limite de lignes doit être un entier positif.")
    lim = re.search(r"\bLIMIT\s+(\d+)\s*$", candidate, re.I)
    if lim:
        candidate = candidate[:lim.start()] + f"LIMIT {min(int(lim.group(1)), max_rows)}"
    elif re.search(r"\bLIMIT\b", candidate, re.I):
        raise ValueError("La clause LIMIT doit comporter un entier.")
    else:
        candidate += f" LIMIT {max_rows}"
    return candidate


def _content(response: Any) -> str:
    value = getattr(response, "content", response)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in value).strip()
    return str(value).strip()


def _engine_from_env() -> Engine:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL n'est pas défini.")
    return create_engine(url, pool_pre_ping=True)


class SQLToolRunner:
    """Exécuteur injectable : modèle LangChain compatible invoke(prompt), moteur ou URL."""
    def __init__(self, llm: Any, engine: Engine | None = None, database_url: str | None = None,
                 max_rows: int = MAX_ROWS, timeout_seconds: int = QUERY_TIMEOUT_SECONDS,
                 correction_attempts: int = 1):
        if engine is None:
            url = database_url or os.getenv("DATABASE_URL")
            if not url:
                raise RuntimeError("Fournir engine/database_url ou définir DATABASE_URL.")
            engine = create_engine(url, pool_pre_ping=True)
        self.llm, self.engine = llm, engine
        self.max_rows, self.timeout_seconds = max_rows, timeout_seconds
        self.correction_attempts = max(0, min(correction_attempts, 2))

    def _generate(self, prompt: str) -> str:
        if hasattr(self.llm, "invoke"):
            return _content(self.llm.invoke(prompt))
        return _content(self.llm(prompt))

    def _execute(self, sql: str) -> list[dict[str, Any]]:
        with self.engine.connect() as conn:
            if conn.dialect.name == "sqlite":
                raw = conn.connection.driver_connection
                raw.execute("PRAGMA query_only = ON")
                import time
                started = time.monotonic()
                raw.set_progress_handler(lambda: 1 if time.monotonic() - started > self.timeout_seconds else 0, 1000)
            with conn.begin():
                if conn.dialect.name == "postgresql":
                    conn.execute(text("SET TRANSACTION READ ONLY"))
                    conn.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": f"{self.timeout_seconds}s"})
                result = conn.execute(text(sql))
                return [dict(row) for row in result.mappings().all()]

    def run(self, question: str) -> str:
        if _out_of_scope(question):
            return OUT_OF_SCOPE_MESSAGE
        prompt = ("Tu génères une requête PostgreSQL en lecture seule pour SportSee. Réponds uniquement par SQL, sans markdown.\n"
                  + SCHEMA_DESCRIPTION + "\n\nExemples:\n" + FEW_SHOT_EXAMPLES + "\n\nQuestion utilisateur: " + question)
        last_error: Exception | None = None
        for attempt in range(self.correction_attempts + 1):
            try:
                generated = self._generate(prompt)
                sql = validate_sql(generated, self.max_rows)
                rows = self._execute(sql)
                return json.dumps({"sql": sql, "rows": rows, "count": len(rows)}, ensure_ascii=False, default=str, separators=(",", ":"))
            except Exception as exc:
                last_error = exc
                if attempt >= self.correction_attempts:
                    break
                prompt = ("La requête précédente a échoué. Corrige-la; retourne uniquement une requête SELECT PostgreSQL sécurisée.\n"
                          f"Question: {question}\nSQL précédent: {generated if 'generated' in locals() else ''}\nErreur: {exc}\n"
                          + SCHEMA_DESCRIPTION + "\nExemples:\n" + FEW_SHOT_EXAMPLES)
        return json.dumps({"error": "Échec de la requête SQL après correction", "detail": str(last_error)},
                          ensure_ascii=False, separators=(",", ":"))


def make_sql_tool(llm: Any | None = None, engine: Engine | None = None,
                  database_url: str | None = None, **runner_options: Any) -> StructuredTool:
    """Construit un tool LangChain; llm injectable ou configuré par variables d'environnement."""
    model = llm or llm_from_env()
    runner = SQLToolRunner(model, engine=engine, database_url=database_url, **runner_options)
    return StructuredTool.from_function(
        func=runner.run,
        name="sportsee_sql_readonly",
        description=("Répond aux questions numériques/statistiques NBA présentes dans les tables SportSee. "
                     "Utiliser pour classements, comparaisons, moyennes et agrégations. "
                     "Ne peut pas répondre aux derniers matchs, à domicile/extérieur ni à une saison absente."),
        args_schema=None,
    )


def llm_from_env() -> Any:
    """Crée un ChatOpenAI ou ChatMistralAI selon SQL_LLM_PROVIDER; aucune clé codée en dur."""
    provider = os.getenv("SQL_LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("SQL_LLM_MODEL")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name or "gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    if provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(model=model_name or "mistral-small-latest", temperature=0, api_key=os.getenv("MISTRAL_API_KEY"))
    raise ValueError("SQL_LLM_PROVIDER doit être 'openai' ou 'mistral'.")


# Nom compatible avec les imports courants du prototype.
sql_tool = make_sql_tool
