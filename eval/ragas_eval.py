from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def run_ragas(question: str, answer: str, contexts: list[str]) -> dict:
    """
    Run RAGAS evaluation metrics on a single QA pair.
    Uses Gemini 2.5 Flash as the judge LLM via OpenAI-compatible endpoint.
    Returns a dict with faithfulness, answer_relevancy, context_precision scores.
    Falls back to zeros on any error (RAGAS is optional — shouldn't crash the API).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from langchain_openai import ChatOpenAI
        from langchain_community.embeddings import HuggingFaceEmbeddings

        llm = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_tokens=4096,
            temperature=0.0,
        )
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        ds = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=llm,
            embeddings=embeddings,
        )

        scores = {k: round(float(v), 3) for k, v in result.items()}
        logger.info("RAGAS scores: %s", scores)
        return scores

    except Exception as exc:
        logger.warning("RAGAS evaluation failed: %s", exc)
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
