"""Exemple autonome de branchement SQL + RAG Reddit dans un agent LangGraph."""
from __future__ import annotations

import os
from typing import Any

from sql_tool import make_sql_tool, llm_from_env

SYSTEM_PROMPT = """Tu es l'assistant SportSee. Réponds toujours en français, clairement et sans extrapoler.
Appelle sportsee_sql_readonly pour les questions quantitatives NBA: meilleurs/top joueurs,
comparaisons, sommes, moyennes, taux et filtres statistiques. Les données SQL sont des agrégats
par joueur/équipe, pas des performances match par match; points est une valeur source, pas
nécessairement une moyenne par match. La saison est inconnue et la table matches est vide.
Pour « derniers matchs », domicile/extérieur ou une saison non disponible, indique explicitement
que la base ne permet pas la réponse. N'invente pas de SQL, chiffre ni saison.
Pour les discussions, opinions, anecdotes et explications présentes dans les publications Reddit,
utilise le retriever RAG textuel fourni (et cite si disponible les sources retournées). N'utilise
pas SQL pour rechercher le contenu de Reddit. Si une question combine statistiques et opinions,
utilise le SQL pour les chiffres et le RAG pour les publications, puis distingue les résultats.
Synthétise la sortie des outils dans une réponse concise et compréhensible en français."""


def build_agent(model: Any | None = None, retriever: Any | None = None,
                sql_tool: Any | None = None, checkpointer: Any | None = None):
    """Retourne un graphe ReAct autonome. Pour le prototype existant: ajouter ce tool à sa liste
    tools et SYSTEM_PROMPT à son prompt système; passer le retriever Reddit existant ici."""
    try:
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        raise RuntimeError("Installer langgraph pour utiliser l'exemple d'agent.") from exc
    model = model or llm_from_env()
    sql = sql_tool or make_sql_tool(llm=model)
    tools = [sql]
    if retriever is not None:
        # Le retriever LangChain est adapté à un outil de recherche de documents.
        from langchain_core.tools import create_retriever_tool
        tools.append(create_retriever_tool(retriever, "recherche_reddit",
                                           "Recherche dans le contenu textuel des publications Reddit indexées."))
    return create_react_agent(model, tools, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)


def run_example(question: str, retriever: Any | None = None) -> Any:
    """Démonstration appelable depuis CLI; DATABASE_URL et fournisseur LLM en environnement."""
    graph = build_agent(retriever=retriever)
    return graph.invoke({"messages": [("user", question)]})


if __name__ == "__main__":
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("Définir DATABASE_URL avant le lancement.")
    result = run_example("Quels sont les cinq meilleurs joueurs en points ?")
    print(result["messages"][-1].content)
