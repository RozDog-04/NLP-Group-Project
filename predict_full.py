import json
from typing import Dict, List

from data_utils import load_hotpot_json
from llm_pipeline import AnswerGenerator, ContextReranker
from llm_query_utils import MistralCompleter
from question_reformulating import QuestionRewriter
from multi_BM25_retrieval import MultiTrajectoryBM25Retriever

def run_full_dev(
    dev_json_path: str,
    output_path: str = "predictions_dev.json",
    top_k_for_answer: int = 10,
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
    # 1) Loads dev data
    data = load_hotpot_json(dev_json_path)

    # 2) Initialise components once
    completer = MistralCompleter()
    rewriter = QuestionRewriter(completer)
    multi_ret = MultiTrajectoryBM25Retriever(
        index_path="data/index/bm25s_index",
        store_path="data/index/bm25_store.pkl",
    )

    answer_gen = AnswerGenerator()
    reranker = ContextReranker()

    predictions: Dict[str, str] = {}

    # 3) Loops over all dev examples
    for idx, sample in enumerate(data):
        if idx >= 10:
            break
        question = sample.get("question", "")
        example_id = sample.get("_id") or sample.get("id")
        if not example_id:
            continue

        # Multi-trajectory retrieval
        traj_results = multi_ret.multi_trajectory_retrieve(
            question=question,
            rewriter=rewriter,
            top_k_per_query=8,
        )

        candidate_answers: List[str] = []
        trajectory_contexts: List[List[str]] = []

        # Processes each trajectory
        for info in traj_results.values():
            docs = info["docs"]
            retrieved_ctxs = [d["text"] for d in docs]

            if not retrieved_ctxs:
                continue

            # Rerank
            if reranker is not None:
                rerank_scores = reranker.score(question, retrieved_ctxs)
                reranked_local = sorted(range(len(retrieved_ctxs)), key=lambda i: rerank_scores[i], reverse=True)
                ranked_ctxs = [retrieved_ctxs[i] for i in reranked_local]
            else:
                ranked_ctxs = retrieved_ctxs

            # Select top-k
            k = min(top_k_for_answer, len(ranked_ctxs))
            selected_ctxs = ranked_ctxs[:k]

            trajectory_contexts.append(selected_ctxs)

            ans = answer_gen.generate_answer(question, selected_ctxs)
            candidate_answers.append(ans)

        # Merge contexts across trajectories (deduplicate) for scoring
        merged_contexts = []
        seen = set()
        for ctx_list in trajectory_contexts:
            for ctx in ctx_list:
                if ctx not in seen:
                    seen.add(ctx)
                    merged_contexts.append(ctx)

        answer_scores = answer_gen.score_candidate_answers(
            question=question,
            contexts=merged_contexts,
            answers=candidate_answers,
        )

        def norm_ans(ans: str) -> str:
            return ans.strip().lower()

        def is_non_empty(ans: str) -> bool:
            return norm_ans(ans) and norm_ans(ans) != "i cannot answer from the given context."

        candidates_idx = list(range(len(candidate_answers)))
        non_empty_idxs = [i for i in candidates_idx if is_non_empty(candidate_answers[i])]

        # Majority vote first
        freq = {}
        for idx_candidate in candidates_idx:
            key = norm_ans(candidate_answers[idx_candidate])
            freq[key] = freq.get(key, []) + [idx_candidate]

        def best_by_frequency(indices: List[int]) -> int:
            if not indices:
                return -1
            groups = {}
            for i in indices:
                key = norm_ans(candidate_answers[i])
                groups.setdefault(key, []).append(i)

            def group_key(item):
                key, idxs = item
                count = len(idxs)
                conf = max((answer_scores[i] if i < len(answer_scores) else 0.0) for i in idxs) if answer_scores else 0.0
                fallback_penalty = 0 if key and key != "i cannot answer from the given context." else -1
                return (fallback_penalty, count, conf)

            best_group_key, best_group_idxs = max(groups.items(), key=group_key)
            if answer_scores:
                return max(best_group_idxs, key=lambda i: answer_scores[i] if i < len(answer_scores) else 0.0)
            return best_group_idxs[0]

        pool = non_empty_idxs if non_empty_idxs else candidates_idx
        best_idx = best_by_frequency(pool)

        if best_idx == -1:
            final_answer = candidate_answers[0] if candidate_answers else ""
        else:
            final_answer = candidate_answers[best_idx]

        predictions[example_id] = final_answer

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
