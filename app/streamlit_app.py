import sys
from pathlib import Path

import streamlit as st

# Project configuration
BASE_DIR = Path(__file__).resolve().parent.parent
RETRIEVAL_DIR = BASE_DIR / "retrieval"

# Allow Python to find the existing retrieval modules
sys.path.insert(0, str(RETRIEVAL_DIR))

st.set_page_config(
    page_title="Spotify Architecture RAG",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 Spotify Architecture Assistant")
st.caption("Hybrid RAG + Knowledge Graph RAG")

st.markdown("---")

st.subheader("Ask a question about the architecture")

question = st.text_input(
    "Your question",
    placeholder="What technologies are used in the frontend?",
)

if st.button("Search and Generate Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            with st.spinner("Searching documents and generating an answer..."):
                from rag_pipeline import generate_answer

                result = generate_answer(question.strip())

            st.subheader("Answer")
            st.write(result["answer"])

            sources = result.get("sources", [])

            if sources:
                st.subheader("Sources")

                for source in sources:
                    st.markdown(
                        f"- **Page:** {source.get('page', 'Unknown')} "
                        f"| **Chunk:** {source.get('chunk_id', 'Unknown')}"
                    )

        except ImportError as exc:
            st.error("Could not import the RAG pipeline.")
            st.code(str(exc))

        except Exception as exc:
            st.error("The RAG request failed.")
            st.code(str(exc))

st.markdown("---")
st.caption(
    "Prototype application — Spotify architecture document knowledge base."
)