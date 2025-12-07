# predict_full.py

import json
from typing import Dict

from data_utils import load_hotpot_json, extract_context_paragraphs
from llm_pipeline import AnswerGenerator, ContextReranker
from predict import DenseRetriever


def run_full_dev(
    dev_json_path: str,
    output_path: str = "predictions_dev.json",
    top_k_for_answer: int = 5,
) -> None:
    """
    Run the RAG pipeline on the entire HotpotQA dev set and write predictions to a JSON file.

    Output format:
        {
            "<example_id>": "<predicted_answer>",
            ...
        }
    where <example_id> is `_id` (or `id` if `_id` is missing) from the dev examples.
    """
    # 1) Load dev data
    data = load_hotpot_json(dev_json_path)

    # 2) Initialise components once
    retriever = DenseRetriever()
    answer_gen = AnswerGenerator()
    reranker = ContextReranker()

    predictions: Dict[str, str] = {}

    # 3) Loop over all dev examples
    for idx, sample in enumerate(data):
        if idx >= 100:        # change this number to process fewer examples for testing
            break

        question = sample.get("question", "")
        example_id = sample.get("_id") or sample.get("id")
        if not example_id:
            # Skip if no ID (shouldn't happen in normal Hotpot dev)
            continue

        # Build all candidate contexts (one per page)
        contexts = extract_context_paragraphs(sample, include_titles=True)

        if not contexts:
            predictions[example_id] = ""
            continue

        # First-stage dense retrieval over all candidates
        dense_indices, dense_scores = retriever.retrieve(
            question, contexts, top_k=len(contexts)
        )

        # LLM-based reranking of those candidates
        dense_ctxs = [contexts[i] for i in dense_indices]
        rerank_scores = reranker.score(question, dense_ctxs)

        # Sort by rerank score (descending)
        reranked_local = sorted(
            range(len(dense_ctxs)),
            key=lambda i: rerank_scores[i],
            reverse=True,
        )
        ranked_indices = [dense_indices[i] for i in reranked_local]

        # Select top-k contexts for answering
        k = min(top_k_for_answer, len(ranked_indices))
        top_indices = ranked_indices[:k]
        selected_ctxs = [contexts[i] for i in top_indices]

        # Generate a short Hotpot-style answer (yes/no or short phrase)
        answer = answer_gen.generate_answer(question, selected_ctxs)

        predictions[example_id] = answer

        # Optional progress print every 100 examples
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(data)} examples")

    # 4) Write predictions to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions for {len(predictions)} examples to {output_path}")


if __name__ == "__main__":
    # Adjust the path if your dev JSON is somewhere else
    run_full_dev(
        dev_json_path="hotpot_dev_distractor_v1.json",
        output_path="predictions_dev.json",
        top_k_for_answer=5,
    )
