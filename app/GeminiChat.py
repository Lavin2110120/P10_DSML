"""Application Streamlit de chat RAG avec Gemini."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import ssl_bypass
from src.config import SEARCH_K
from src.rag.vector_store import VectorStoreManager
from src.llm.gemini_client import GeminiClient
import streamlit as st

st.set_page_config(page_title="Assistant RAG Gemini", page_icon="🏀", layout="wide")

@st.cache_resource
def init_rag():
    vsm = VectorStoreManager()
    vsm.load()
    client = GeminiClient()
    return vsm, client

st.title("🏀 Assistant RAG - Analyse NBA & Forums")

try:
    vsm, client = init_rag()
    st.sidebar.success("Index FAISS (Gemini) connecté ✅")
except Exception as e:
    st.sidebar.error(f"Erreur d'initialisation : {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération..."):
            docs = vsm.search(prompt, top_k=SEARCH_K)
            context = "\n\n---\n\n".join([d.get("content", "") for d in docs])
            
            system_instruction = (
                "Tu es un assistant expert NBA. Réponds à la question en t'appuyant rigoureusement "
                "sur le contexte fourni. Si le contexte ne contient pas l'info, indique-le clairement.\n\n"
                f"CONTEXTE :\n{context}"
            )
            response = client.generate(prompt=prompt, system_prompt=system_instruction)
            st.markdown(response)
            
            with st.expander("Sources consultées"):
                for idx, doc in enumerate(docs, 1):
                    st.write(f"**Source {idx} :** {doc.get('metadata', {}).get('source', 'Inconnue')}")
                    st.caption(doc.get("content", "")[:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": response})