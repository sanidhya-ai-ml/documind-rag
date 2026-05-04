from __future__ import annotations

import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None

_SYSTEM = (
    "You are a precise document assistant. Answer the user's question using ONLY "
    "the provided context passages. If the context does not contain enough information "
    "to answer, say so clearly. Be concise and cite relevant details."
)


def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_tokens=8192,
            temperature=0.2,
        )
    return _llm


def generate(question: str, contexts: list[str]) -> str:
    if not contexts:
        return "No relevant context found in the documents to answer this question."

    context_block = "\n\n".join(
        f"[Context {i + 1}]\n{ctx}" for i, ctx in enumerate(contexts)
    )
    user_msg = f"Context passages:\n{context_block}\n\nQuestion: {question}"

    try:
        resp = _get_llm().invoke([
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        return (resp.content or "").strip()
    except Exception as exc:
        logger.warning("LLM generation failed: %s", exc)
        return "Answer generation failed. Please try again."
