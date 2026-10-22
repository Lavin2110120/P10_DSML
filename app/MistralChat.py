# app/MistralChat.py
import sys
from pathlib import Path

# Ajouter le dossier src/ au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import logging
from typing import Any, Dict, List

import streamlit as st
import faiss
import numpy as np

# ============================================================
# CONFIGURATION DES CHEMINS (adapté à ton arborescence)
# ============================================================
SRC_DIR = SRC_PATH
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"

# ============================================================
# IMPORTS DEPUIS src/ (maintenant accessibles)
# ============================================================
from llm.mistral_client import MistralClient
from rag.vector_store import VectorStoreManager
from rag.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K,
    APP_TITLE,
    NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# ============================================================
# CONFIGURATION LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏀",
    layout="wide"
)

st.title(f"🏀 {APP_TITLE}")
st.markdown(f"Assistant IA pour l'analyse des matchs NBA - {NAME}")

# ============================================================
# INITIALISATION DES COMPOSANTS
# ============================================================

def load_vector_store():
    """Charge l'index FAISS une seule fois dans l'état de session."""
    if "vector_store" not in st.session_state:
        try:
            vector_manager = VectorStoreManager()
            # Le constructeur charge déjà l'index FAISS et les chunks.
            logger.info(f"Index FAISS chargé depuis {VECTOR_DB_DIR}")
            st.session_state.vector_store = vector_manager
        except FileNotFoundError:
            st.error(f"❌ Fichiers d'index non trouvés dans {VECTOR_DB_DIR}")
            st.info("💡 Veuillez d'abord lancer l'indexation (étape 1) pour créer les fichiers dans vector_db/")
            logger.exception("FileNotFoundError lors du chargement du VectorStoreManager")
            st.session_state.vector_store = None
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement de l'index: {e}")
            logger.exception("Erreur lors du chargement de l'index")
            st.session_state.vector_store = None
    return st.session_state.vector_store


def load_mistral_client():
    """Initialise le client Mistral une seule fois dans l'état de session."""
    if "mistral_client" not in st.session_state:
        if not MISTRAL_API_KEY:
            st.error("❌ Clé API Mistral non trouvée dans .env (MISTRAL_API_KEY)")
            st.session_state.mistral_client = None
        else:
            try:
                st.session_state.mistral_client = MistralClient(
                    api_key=MISTRAL_API_KEY,
                    model=MODEL_NAME
                )
                logger.info("Client Mistral initialisé")
            except Exception as e:
                st.error(f"❌ Erreur lors de l'initialisation du client Mistral: {e}")
                logger.exception("Erreur initialisation client Mistral")
                st.session_state.mistral_client = None
    return st.session_state.mistral_client


# Le bouton est traité avant le chargement : il permet de recréer les deux
# composants sans toucher à l'historique de conversation.
with st.sidebar:
    if st.button("🔄 Réinitialiser"):
        st.session_state.pop("vector_store", None)
        st.session_state.pop("mistral_client", None)
        st.rerun()

try:
    load_vector_store()
    load_mistral_client()
except Exception as e:
    st.error(f"Erreur d'initialisation: {e}")
    st.stop()

if st.session_state.vector_store is None or st.session_state.mistral_client is None:
    st.stop()

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def retrieve_context(query: str, k: int = SEARCH_K) -> List[Dict[str, Any]]:
    """
    Récupère les chunks pertinents depuis l'index FAISS.
    """
    try:
        results = st.session_state.vector_store.search(query, k=k)
        return results
    except Exception as e:
        logger.error(f"Erreur recherche: {e}")
        return []


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formate les chunks récupérés pour le prompt.
    """
    if not chunks:
        return "Aucun contexte pertinent trouvé."

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get('metadata', {}).get('source', 'Inconnu')
        content = chunk.get('content', chunk.get('text', ''))
        score = chunk.get('score', 0)

        context_parts.append(
            f"[Document {i} - Source: {source} - Score: {score:.3f}]\n{content}\n"
        )

    return "\n---\n".join(context_parts)


def generate_response(query: str, context: str) -> str:
    """
    Génère une réponse avec Mistral en utilisant le contexte RAG.
    """
    system_prompt = """Tu es un assistant expert en analyse de matchs NBA.
Tu réponds aux questions en te basant UNIQUEMENT sur le contexte fourni.
Si l'information n'est pas dans le contexte, dis-le clairement.
Sois précis, factuel et cite tes sources quand c'est pertinent."""

    user_prompt = f"""Contexte disponible:
{context}

Question: {query}

Réponse détaillée basée sur le contexte ci-dessus:"""

    try:
        response = st.session_state.mistral_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.3,
            max_tokens=1000
        )
        return response
    except Exception as e:
        logger.error(f"Erreur génération Mistral: {e}")
        return f"❌ Erreur lors de la génération de la réponse: {str(e)}"

# ============================================================
# INTERFACE STREAMLIT
# ============================================================

# Sidebar avec infos et statistiques
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown(f"""
    **Modèle:** {MODEL_NAME}

    **Base vectorielle:** {VECTOR_DB_DIR.name}/

    **Documents indexés:** {len(st.session_state.vector_store.document_chunks) if hasattr(st.session_state.vector_store, 'document_chunks') else 'N/A'}

    **Données sources:** {DATA_RAW_DIR.name}/
    """)

    st.markdown("---")
    st.header("📊 Statistiques")
    if 'messages' in st.session_state:
        st.metric("Messages échangés", len(st.session_state.messages))

    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.rerun()

# Initialisation de l'historique
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"Bonjour ! Je suis {NAME}, votre assistant d'analyse NBA. Posez-moi vos questions sur les matchs, statistiques ou commentaires des rencontres."
    })

# Affichage de l'historique
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input utilisateur
if prompt := st.chat_input("Posez votre question sur les matchs NBA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Recherche dans les documents..."):
            chunks = retrieve_context(prompt, k=SEARCH_K)

            if not chunks:
                st.warning("⚠️ Aucun document pertinent trouvé pour cette question.")
                response = "Je n'ai pas trouvé d'information pertinente dans ma base de connaissances pour répondre à cette question."
            else:
                with st.expander(f"📚 {len(chunks)} sources trouvées", expanded=False):
                    for i, chunk in enumerate(chunks, 1):
                        source = chunk.get('metadata', {}).get('source', 'Inconnu')
                        score = chunk.get('score', 0)
                        content_preview = chunk.get('content', chunk.get('text', ''))[:200] + "..."
                        st.markdown(f"**{i}. {source}** (score: {score:.3f})")
                        st.caption(content_preview)

                with st.spinner("🤖 Génération de la réponse avec Mistral..."):
                    context_text = format_context(chunks)
                    response = generate_response(prompt, context_text)

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.markdown("---")
st.caption(f"Powered by Mistral AI & FAISS | {APP_TITLE} v1.0")
