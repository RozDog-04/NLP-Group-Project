import numpy as np
from sentence_transformers import SentenceTransformer

from data_utils import load_hotpot_json, extract_context_paragraphs
from llm_pipeline import AnswerGenerator, ContextReranker
from llm_query_utils import MistralCompleter
from question_reformulating import QuestionRewriter
from multi_BM25_retrieval import MultiTrajectoryBM25Retriever

def run_simple_pipeline(dev_json_path: str, n_samples: int = 3, top_k_for_answer: int = 5):
    data = load_hotpot_json(dev_json_path)
    
    # Initializes components
    completer = MistralCompleter()
    rewriter = QuestionRewriter(completer)
    multi_ret = MultiTrajectoryBM25Retriever(
        index_path="data/index/bm25s_index",
        store_path="data/index/bm25_store.pkl",
    )
    
    answer_gen = AnswerGenerator()
    reranker = ContextReranker()

    for sample in data[:n_samples]:
        question = sample.get("question", "")
        gold = sample.get("answer", "")

        print(f"Question: {question}")
        print(f"Original Answer: {gold}")
        print("-" * 40)

        # Multi-trajectory retrieval
        # Returns dict with {"original", "rewrite", "decomp", "entity"} keys
        traj_results = multi_ret.multi_trajectory_retrieve(
            question=question,
            rewriter=rewriter,
            top_k_per_query=8,
        )

        candidate_answers = []
        trajectory_contexts = []
        trajectory_names = []

        # Processes each trajectory
        for traj_name, info in traj_results.items():
            queries = info["queries"]
            docs = info["docs"]
            
            # Extract text
            retrieved_ctxs = [d["text"] for d in docs]
            
            if not retrieved_ctxs:
                print(f"Trajectory: {traj_name} (No docs found)")
                continue

            # Rerank with ContextReranker if available
            if reranker is not None:
                rerank_scores = reranker.score(question, retrieved_ctxs)
                reranked_local = sorted(range(len(retrieved_ctxs)), key=lambda i: rerank_scores[i], reverse=True)
                ranked_ctxs = [retrieved_ctxs[i] for i in reranked_local]
            else:
                ranked_ctxs = retrieved_ctxs

            # Selects top-k
            top_k = min(top_k_for_answer, len(ranked_ctxs))
            selected_ctxs = ranked_ctxs[:top_k]

            # Generates answer
            answer = answer_gen.generate_answer(question, selected_ctxs)

            candidate_answers.append(answer)
            trajectory_contexts.append(selected_ctxs)
            trajectory_names.append(traj_name)

            print(f"Trajectory: {traj_name}")
            print(f"  Queries: {queries}")
            print(f"  Generated Answer: {answer}")

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

        def is_non_empty(ans: str) -> bool:
            return ans.strip() and ans.strip().lower() != "i cannot answer from the given context."

        candidates_idx = list(range(len(candidate_answers)))
        non_empty_idxs = [i for i in candidates_idx if is_non_empty(candidate_answers[i])]

        # Majority vote first: prefer answers that overlap across trajectories.
        def norm_ans(ans: str) -> str:
            return ans.strip().lower()

        freq = {}
        for idx in candidates_idx:
            key = norm_ans(candidate_answers[idx])
            freq[key] = freq.get(key, []) + [idx]

        # Pick the answer with the highest count (excluding fallback if possible)
        def is_fallback(ans: str) -> bool:
            return norm_ans(ans) == "i cannot answer from the given context."

        def best_by_frequency(indices: list[int]) -> int:
            if not indices:
                return -1
            # group by normalized answer and count
            groups = {}
            for i in indices:
                key = norm_ans(candidate_answers[i])
                groups.setdefault(key, []).append(i)
            # prefer non-fallback groups, then higher count, then highest confidence
            def group_key(item):
                key, idxs = item
                count = len(idxs)
                conf = max((answer_scores[i] if i < len(answer_scores) else 0.0) for i in idxs) if answer_scores else 0.0
                fallback_penalty = 0 if key and key != "i cannot answer from the given context." else -1
                return (fallback_penalty, count, conf)
            best_group_key, best_group_idxs = max(groups.items(), key=group_key)
            # within the best group, pick the idx with highest confidence
            if answer_scores:
                return max(best_group_idxs, key=lambda i: answer_scores[i] if i < len(answer_scores) else 0.0)
            return best_group_idxs[0]

        pool = non_empty_idxs if non_empty_idxs else candidates_idx
        best_idx = best_by_frequency(pool)

        if best_idx == -1:
            final_answer = candidate_answers[0] if candidate_answers else ""
            best_idx = 0 if candidate_answers else -1
        else:
            final_answer = candidate_answers[best_idx]

        print("Candidate answers and confidences:")
        for i, (ans, conf) in enumerate(zip(candidate_answers, answer_scores)):
            name = trajectory_names[i] if i < len(trajectory_names) else f"traj_{i}"
            print(f"  [{i}] ({name}) {ans!r} (conf={conf:.2f})")

        print("Chosen Contexts:")
        if 0 <= best_idx < len(trajectory_contexts):
            for ctx in trajectory_contexts[best_idx]:
                print(f"- {ctx[:100]}...")

        print(f"Chosen final answer: {final_answer}")
        print("=" * 60)


if __name__ == "__main__":
    run_simple_pipeline("hotpot_dev_distractor_v1.json", n_samples=5)
