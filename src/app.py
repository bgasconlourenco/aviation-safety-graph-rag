"""Phase 9: Streamlit demo — Aviation Safety RAG Assistant."""

import sys
from pathlib import Path

# ensure project root is on sys.path when launched via `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import os

import streamlit as st
from dotenv import load_dotenv

from src.graph_load import connect
from src.rag import ask

load_dotenv()

st.set_page_config(
    page_title="Aviation Safety Assistant",
    page_icon="🚁",
    layout="wide",
)


# --- Cached resources (initialised once per session) ---

@st.cache_resource(show_spinner="Connecting to Neo4j…")
def get_driver():
    return connect()


@st.cache_resource(show_spinner="Loading embedding model…")
def get_model():
    if os.getenv("HF_API_TOKEN"):
        return None  # embeddings handled via HF Inference API
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    from src.embed import MODEL_NAME  # noqa: PLC0415
    return SentenceTransformer(MODEL_NAME)


# --- Helper ---

def render_sources(hits: list[dict], enrichment: dict[str, dict]) -> None:
    with st.expander(f"Sources — {len(hits)} incidents retrieved", expanded=False):
        for hit in hits:
            acn = hit["acn"]
            meta = enrichment.get(acn, {})

            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**ACN {acn}** — {hit.get('synopsis') or '*(no synopsis)*'}")
            with col2:
                st.metric("Similarity", f"{hit['score']:.0%}")

            tags = []
            for label, key in [
                ("Mission", "mission"),
                ("Phase", "flight_phase"),
                ("Conditions", "flight_conditions"),
                ("Problem", "primary_problem"),
            ]:
                if meta.get(key):
                    tags.append(f"{label}: {meta[key]}")
            if meta.get("anomalies"):
                tags.append(f"Anomalies: {', '.join(meta['anomalies'])}")
            if tags:
                st.caption(" · ".join(tags))
            if hit.get("date"):
                st.caption(f"Date: {hit['date']}")

            st.divider()


# --- Layout ---

st.title("🚁 Aviation Safety RAG Assistant")
st.caption(
    "Ask questions about NASA ASRS helicopter incident reports (1988–2026). "
    "Answers are grounded in real incident data retrieved from a Neo4j knowledge graph."
)

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Incidents to retrieve", min_value=1, max_value=20, value=5)

if "history" not in st.session_state:
    st.session_state.history = []

# Render conversation history
for q, result in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        render_sources(result["sources"], result["enrichment"])

# Handle new input
question = st.chat_input("Ask a question about helicopter safety…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching incidents and generating answer…"):
            try:
                result = ask(get_driver(), question, model=get_model(), top_k=top_k)
            except RuntimeError as exc:
                st.error(str(exc))
                st.stop()

        st.markdown(result["answer"])
        render_sources(result["sources"], result["enrichment"])

    st.session_state.history.append((question, result))
