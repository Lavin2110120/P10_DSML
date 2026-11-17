import json
import re

import pytest
from sqlalchemy import create_engine, text
from langchain_core.messages import AIMessage

from sql_tool import SQLToolRunner, validate_sql, OUT_OF_SCOPE_MESSAGE


class FakeLLM:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return AIMessage(content=answer)


@pytest.fixture
def engine():
    e = create_engine("sqlite:///:memory:")
    with e.begin() as c:
        c.execute(text("CREATE TABLE players (player_id INTEGER PRIMARY KEY, source_name TEXT NOT NULL)"))
        c.execute(text("CREATE TABLE stats (stats_id INTEGER PRIMARY KEY, player_id INTEGER, team_code TEXT, dataset_key TEXT, gp INTEGER, points INTEGER, rebounds INTEGER, three_point_pct REAL, assists INTEGER, assist_to_turnover REAL)"))
        c.execute(text("INSERT INTO players VALUES (1,'Joueur A'),(2,'Joueur B'),(3,'Joueur C')"))
        c.execute(text("INSERT INTO stats VALUES (1,1,'BOS','sample',20,100,50,40.0,30,2.0),(2,2,'NYK','sample',10,90,70,35.0,20,1.0),(3,3,'BOS','sample',5,110,60,45.0,40,3.0)"))
    yield e
    e.dispose()


def test_rejects_unsafe_and_multistatements():
    for sql in ["DROP TABLE players", "SELECT * FROM players; UPDATE players SET source_name='x'", "UPDATE players SET source_name='x'", "SELECT * FROM players; SELECT * FROM stats"]:
        with pytest.raises(ValueError):
            validate_sql(sql)


def test_validate_forces_and_caps_limit():
    assert validate_sql("SELECT * FROM players") == "SELECT * FROM players LIMIT 100"
    assert validate_sql("SELECT * FROM players LIMIT 900", max_rows=7).endswith("LIMIT 7")
    assert validate_sql("SELECT * FROM players LIMIT 5", max_rows=7).endswith("LIMIT 5")


def test_executes_and_returns_compact_json(engine):
    llm = FakeLLM(["SELECT source_name FROM players ORDER BY player_id"])
    result = SQLToolRunner(llm, engine=engine).run("Liste les joueurs")
    payload = json.loads(result)
    assert payload["count"] == 3
    assert payload["rows"][0]["source_name"] == "Joueur A"
    assert payload["sql"].endswith("LIMIT 100")
    assert len(result) < 500


def test_correction_after_sql_error(engine):
    llm = FakeLLM(["SELECT colonne_inconnue FROM players", "SELECT source_name FROM players LIMIT 1"])
    result = json.loads(SQLToolRunner(llm, engine=engine, correction_attempts=1).run("Un joueur"))
    assert result["rows"][0]["source_name"] == "Joueur A"
    assert len(llm.prompts) == 2
    assert "Erreur" in llm.prompts[1]


def test_out_of_scope_returns_explicit_message_without_llm(engine):
    llm = FakeLLM([])
    result = SQLToolRunner(llm, engine=engine).run("Quels sont ses 5 derniers matchs à domicile ?")
    assert "ne contient pas de statistiques par match" in result
    assert OUT_OF_SCOPE_MESSAGE == result
    assert llm.prompts == []


def test_sqlite_connection_is_readonly(engine):
    runner = SQLToolRunner(FakeLLM(["SELECT * FROM players"]), engine=engine)
    assert json.loads(runner.run("Liste"))["count"] == 3
    with pytest.raises(Exception):
        runner._execute("DELETE FROM players")


def test_fewshot_tables_and_qualified_columns_match_real_schema():
    from pathlib import Path
    import sql_tool as module
    schema_text = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    schema_tables = {}
    for match in re.finditer(r"CREATE TABLE IF NOT EXISTS\s+(\w+)\s*\((.*?)\n\);", schema_text, re.S | re.I):
        table, body = match.groups()
        columns = set()
        for line in body.splitlines():
            line = line.strip()
            col = re.match(r"(\w+)\s+", line)
            if col and not line.upper().startswith(("CONSTRAINT ", "PRIMARY ", "UNIQUE ", "CHECK ", "FOREIGN ")):
                columns.add(col.group(1).lower())
        schema_tables[table.lower()] = columns

    examples = re.findall(r"SQL:\s*(SELECT.*?)(?=\n\s*\n|$)", module.FEW_SHOT_EXAMPLES, re.S)
    assert len(examples) == 8
    for sql in examples:
        relations = set(re.findall(r"\b(?:FROM|JOIN)\s+(\w+)", sql, re.I))
        assert {table.lower() for table in relations} <= schema_tables.keys()
        aliases = {alias.lower(): table.lower() for table, alias in re.findall(r"\b(?:FROM|JOIN)\s+(\w+)\s+(\w+)", sql, re.I)}
        for alias, col in re.findall(r"\b([a-z])\.(\w+)", sql):
            table = aliases.get(alias.lower())
            if table:
                assert col.lower() in schema_tables[table], f"{alias}.{col} absent dans schema.sql"
