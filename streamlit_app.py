import os
import time

import httpx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8003")

st.set_page_config(page_title="DocuMind RAG", page_icon="📄", layout="wide")
st.title("DocuMind RAG")
st.caption("Hybrid retrieval · BM25 + Vector + CrossEncoder reranking · RAGAS evaluation")

tab1, tab2 = st.tabs(["Upload & Query", "RAGAS Dashboard"])

# ── Tab 1: Upload & Query ──────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Ingest PDF")
        uploaded = st.file_uploader("Choose a PDF", type=["pdf"])
        collection = st.text_input("Collection name", value="my_docs")

        if st.button("Upload & Index", disabled=uploaded is None):
            with st.spinner("Uploading and indexing…"):
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                data = {"collection_name": collection}
                resp = httpx.post(f"{API_URL}/ingest", files=files, data=data, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    task_id = result["task_id"]
                    st.session_state["ingest_task_id"] = task_id
                    st.session_state["collection"] = collection
                    st.success(f"Queued! task_id: `{task_id}`")
                else:
                    st.error(f"Upload failed: {resp.text}")

        if "ingest_task_id" in st.session_state:
            if st.button("Check ingestion status"):
                resp = httpx.get(f"{API_URL}/ingest/{st.session_state['ingest_task_id']}", timeout=10)
                if resp.status_code == 200:
                    status_data = resp.json()
                    status = status_data["status"]
                    if status == "complete":
                        st.success(f"Complete — {status_data['chunk_count']} chunks indexed")
                    elif status == "failed":
                        st.error(f"Failed: {status_data['error']}")
                    else:
                        st.info(f"Status: {status}")

    with col2:
        st.subheader("Query Documents")
        query_collection = st.text_input(
            "Collection to query",
            value=st.session_state.get("collection", "my_docs"),
        )
        question = st.text_area("Your question", height=100, placeholder="What is the main conclusion of this document?")
        top_k = st.slider("Top-K contexts", min_value=1, max_value=10, value=5)

        if st.button("Ask", disabled=not question.strip()):
            with st.spinner("Retrieving and generating…"):
                resp = httpx.post(
                    f"{API_URL}/query",
                    json={"question": question, "collection_name": query_collection, "top_k": top_k},
                    timeout=60,
                )
                if resp.status_code == 200:
                    result = resp.json()
                    st.markdown("### Answer")
                    st.write(result["answer"])

                    with st.expander(f"Retrieved Contexts ({len(result['contexts'])} passages)"):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"**[{i}]** {ctx}")
                            st.divider()

                    st.info("RAGAS evaluation running in background (~10s). Check the RAGAS Dashboard tab.")
                else:
                    st.error(f"Query failed: {resp.text}")

# ── Tab 2: RAGAS Dashboard ─────────────────────────────────────────────────────
with tab2:
    st.subheader("RAGAS Evaluation Dashboard")
    st.caption("Metrics update automatically after each query (faithfulness, answer relevancy, context precision)")

    if st.button("Refresh"):
        st.rerun()

    resp = httpx.get(f"{API_URL}/evaluation", timeout=10)
    if resp.status_code == 200:
        evals = resp.json()
        if not evals:
            st.info("No evaluations yet. Run a query in Tab 1 and wait ~10s.")
        else:
            df = pd.DataFrame(evals)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Avg Faithfulness", f"{df['faithfulness'].mean():.2f}")
            col_b.metric("Avg Answer Relevancy", f"{df['answer_relevancy'].mean():.2f}")
            col_c.metric("Avg Context Precision", f"{df['context_precision'].mean():.2f}")

            st.markdown("### Metrics over queries")
            chart_df = df[["faithfulness", "answer_relevancy", "context_precision"]].reset_index(drop=True)
            st.line_chart(chart_df)

            st.markdown("### Evaluation history")
            display_df = df[["timestamp", "question", "faithfulness", "answer_relevancy", "context_precision"]].copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%H:%M:%S")
            display_df["question"] = display_df["question"].str[:80]
            st.dataframe(display_df, use_container_width=True)
    else:
        st.error(f"Could not reach API at {API_URL}")
