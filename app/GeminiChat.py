"""Application Streamlit de chat RAG Gemini avec réponse Pydantic validée."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import SEARCH_K
from src.llm.gemini_client import GeminiClient
from src.llm.structured_rag import generate_structured_answer
from src.models import RAGAnswer
from src.observability import configure
from src.rag.vector_store import VectorStoreManager

configure()
st.set_page_config(page_title="Assistant RAG Gemini", page_icon="🏀", layout="wide")


@st.cache_resource
def init_rag() -> tuple[VectorStoreManager, GeminiClient]:
    vector_store = VectorStoreManager()
    vector_store.load()
    return vector_store, GeminiClient()


st.title("🏀 Assistant RAG - Analyse NBA & Forums")
try:
    vsm, client = init_rag()
    st.sidebar.success("Index FAISS (Gemini) connecté ✅")
except Exception as exc:
    st.sidebar.error(f"Erreur d'initialisation : {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération..."):
            documents = vsm.search(prompt, top_k=SEARCH_K)
            context = "\n\n---\n\n".join(
                doc.get("text") or doc.get("content") or doc.get("page_content") or ""
                for doc in documents
            )
            try:
                answer = generate_structured_answer(prompt, context, documents, client)
                answer = RAGAnswer.model_validate(answer)
            except Exception:
                # Last-resort fallback for structured-RAG integration/runtime failures.
                raw_answer = client.generate_content(
                    prompt,
                    system_prompt=(
                        "Réponds en français uniquement à partir du contexte fourni. "
                        "Si l'information manque, dis-le clairement.\n\n"
                        f"CONTEXTE :\n{context}"
                    ),
                )
                answer = RAGAnswer(
                    answer=raw_answer.strip() or "Je ne dispose pas d’informations suffisantes dans le contexte.",
                    sources=[],
                    grounded=False,
                    abstained=True,
                )
            st.markdown(answer.answer)
            with st.expander("Sources consultées"):
                for index, source in enumerate(answer.sources, 1):
                    st.write(
                        f"**Source {index} :** "
                        f"{source.source or source.metadata.get('source', 'Inconnue')}"
                    )
                    st.caption(source.text[:300] + ("..." if len(source.text) > 300 else ""))

    st.session_state.messages.append({"role": "assistant", "content": answer.answer})
