from __future__ import annotations

import logging
import os
import pickle

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

_embeddings: HuggingFaceEmbeddings | None = None
_reranker: CrossEncoder | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        logger.info("Loading CrossEncoder reranker")
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def _chroma_search(query: str, collection_name: str, k: int = 10) -> list[str]:
    try:
        db = Chroma(
            collection_name=collection_name,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        results = db.similarity_search(query, k=k)
        return [r.page_content for r in results]
    except Exception as exc:
        logger.warning("ChromaDB search failed: %s", exc)
        return []


def _bm25_search(query: str, collection_name: str, k: int = 10) -> list[str]:
    bm25_path = os.path.join(CHROMA_DIR, f"{collection_name}_bm25.pkl")
    try:
        with open(bm25_path, "rb") as f:
            bm25, texts = pickle.load(f)
        tokens = query.split()
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [texts[i] for i in top_indices]
    except Exception as exc:
        logger.warning("BM25 search failed: %s", exc)
        return []


def hybrid_retrieve(query: str, collection_name: str, top_k: int = 5) -> list[str]:
    dense = _chroma_search(query, collection_name, k=10)
    sparse = _bm25_search(query, collection_name, k=10)

    seen: dict[str, None] = {}
    for text in dense + sparse:
        seen[text] = None
    candidates = list(seen.keys())[:15]

    if not candidates:
        logger.warning("No candidates retrieved for query: %s", query[:60])
        return []

    reranker = _get_reranker()
    scores = reranker.predict([(query, c) for c in candidates])
    ranked = [c for _, c in sorted(zip(scores, candidates), reverse=True)]
    logger.info("Retrieved %d candidates, returning top %d after reranking", len(candidates), top_k)
    return ranked[:top_k]
