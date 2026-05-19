"""
Batch RAGAS evaluation script for DocuMind RAG.

Usage:
    python eval/run_batch_eval.py \
        --collection research \
        --api-url http://localhost:8003 \
        --output eval/results/eval_results.json

Requires:
    - DocuMind RAG API running (docker compose up OR local uvicorn)
    - A PDF already ingested into the target collection
    - GEMINI_API_KEY set in the environment
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from datasets import Dataset
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

QUESTIONS_PATH = Path(__file__).parent / "sample_questions.json"
DEFAULT_OUTPUT = Path(__file__).parent / "results" / "eval_results.json"


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def query_rag(api_url: str, collection: str, question: str, top_k: int = 5) -> tuple[str, list[str]]:
    """POST /query and return (answer, contexts)."""
    resp = httpx.post(
        f"{api_url}/query",
        json={"question": question, "collection_name": collection, "top_k": top_k},
        timeout=60.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data["contexts"]


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=4096,
        temperature=0.0,
    )


def run_batch(
    api_url: str,
    collection: str,
    questions: list[dict],
    output_path: Path,
) -> None:
    logger.info("Querying RAG API for %d questions (collection=%s)", len(questions), collection)

    rows: list[dict] = []
    failed = 0

    for i, item in enumerate(questions, 1):
        q = item["question"]
        gt = item.get("ground_truth", "")
        logger.info("[%d/%d] %s", i, len(questions), q[:80])
        try:
            answer, contexts = query_rag(api_url, collection, q)
        except Exception as exc:
            logger.warning("  Query failed: %s", exc)
            failed += 1
            continue

        rows.append({
            "question": q,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt,
        })
        logger.info("  Retrieved %d context chunks, answer length %d chars", len(contexts), len(answer))

    if not rows:
        logger.error("No successful queries — aborting evaluation.")
        sys.exit(1)

    logger.info("Running RAGAS on %d QA pairs (this calls Gemini as judge — may take 1-3 min)…", len(rows))

    ds = Dataset.from_dict({
        "question": [r["question"] for r in rows],
        "answer": [r["answer"] for r in rows],
        "contexts": [r["contexts"] for r in rows],
        "ground_truth": [r["ground_truth"] for r in rows],
    })

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=build_llm(),
        embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
    )

    scores_df = result.to_pandas()

    per_question: list[dict] = []
    for i, row in enumerate(rows):
        per_question.append({
            "question": row["question"],
            "answer": row["answer"][:300] + ("…" if len(row["answer"]) > 300 else ""),
            "contexts_retrieved": len(row["contexts"]),
            "faithfulness": round(float(scores_df.iloc[i]["faithfulness"]), 3),
            "answer_relevancy": round(float(scores_df.iloc[i]["answer_relevancy"]), 3),
            "context_precision": round(float(scores_df.iloc[i]["context_precision"]), 3),
        })

    avg_faith = round(float(scores_df["faithfulness"].mean()), 3)
    avg_rel = round(float(scores_df["answer_relevancy"].mean()), 3)
    avg_prec = round(float(scores_df["context_precision"].mean()), 3)

    output = {
        "run_at": datetime.now(tz=timezone.utc).isoformat(),
        "collection": collection,
        "api_url": api_url,
        "total_questions": len(questions),
        "evaluated": len(rows),
        "failed": failed,
        "averages": {
            "faithfulness": avg_faith,
            "answer_relevancy": avg_rel,
            "context_precision": avg_prec,
        },
        "per_question": per_question,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("Results saved → %s", output_path)

    print("\n" + "=" * 60)
    print("RAGAS Evaluation Summary")
    print("=" * 60)
    print(f"  Evaluated : {len(rows)} / {len(questions)} questions")
    print(f"  Faithfulness      : {avg_faith:.3f}")
    print(f"  Answer Relevancy  : {avg_rel:.3f}")
    print(f"  Context Precision : {avg_prec:.3f}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation for DocuMind RAG")
    parser.add_argument("--collection", default="research", help="ChromaDB collection name")
    parser.add_argument("--api-url", default="http://localhost:8003", help="DocuMind API base URL")
    parser.add_argument("--questions", default=str(QUESTIONS_PATH), help="Path to questions JSON")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path for results JSON")
    args = parser.parse_args()

    if "GEMINI_API_KEY" not in os.environ:
        sys.exit("Error: GEMINI_API_KEY environment variable not set.")

    questions = load_questions(Path(args.questions))
    logger.info("Loaded %d questions from %s", len(questions), args.questions)

    run_batch(
        api_url=args.api_url,
        collection=args.collection,
        questions=questions,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
