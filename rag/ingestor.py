from __future__ import annotations

import logging
import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")

_embeddings: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL)
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def _bm25_path(collection_name: str) -> str:
    return os.path.join(CHROMA_DIR, f"{collection_name}_bm25.pkl")


def ingest(file_path: str, collection_name: str) -> int:
    logger.info("Loading PDF: %s", file_path)
    docs = PyPDFLoader(file_path).load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    texts = [c.page_content for c in chunks]
    logger.info("Split into %d chunks", len(chunks))

    os.makedirs(CHROMA_DIR, exist_ok=True)
    embeddings = _get_embeddings()
    Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )
    logger.info("ChromaDB collection '%s' saved", collection_name)

    bm25 = BM25Okapi([t.split() for t in texts])
    with open(_bm25_path(collection_name), "wb") as f:
        pickle.dump((bm25, texts), f)
    logger.info("BM25 index saved for '%s'", collection_name)

    return len(chunks)
