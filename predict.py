import numpy as np
from sentence_transformers import SentenceTransformer

from data_utils import load_hotpot_json, extract_context_paragraphs
from llm_pipeline import AnswerGenerator, ContextReranker, QueryRewriter
from BM25S_retrieval import BM25Retriever


def run_simple_pipeline(dev_json_path: str, n_samples: int = 3, top_k_for_answer: int = 5):
    data = load_hotpot_json(dev_json_path)

    index_path = "data/index/bm25s_index"
    store_path = "data/index/bm25_store.pkl"
    retriever = BM25Retriever(index_path=index_path, store_path=store_path)

    answer_gen = AnswerGenerator()
    reranker = ContextReranker()
    query_rewriter = QueryRewriter()

    for sample in data[:n_samples]:
        question = sample.get("question", "")
        gold = sample.get("answer", "")

        # Build two trajectories: original question and an entity-focused rewrite
        queries = [question]
        entity_query = query_rewriter.rewrite_entity_focused(question)
        if entity_query and entity_query not in queries:
            queries.append(entity_query)

        trajectory_contexts = []
        candidate_answers = []

        for q in queries:
            results, scores = retriever.retrieve(q, top_k=10)
            retrieved_ctxs = [r["text"] for r in results]

            if reranker is not None:
                rerank_scores = reranker.score(question, retrieved_ctxs)
                reranked_local = sorted(range(len(retrieved_ctxs)), key=lambda i: rerank_scores[i], reverse=True)
                ranked_ctxs = [retrieved_ctxs[i] for i in reranked_local]
            else:
                ranked_ctxs = retrieved_ctxs

            top_k = min(top_k_for_answer, len(ranked_ctxs))
            selected_ctxs = ranked_ctxs[:top_k]

            trajectory_contexts.append(selected_ctxs)

            answer = answer_gen.generate_answer(question, selected_ctxs)
            candidate_answers.append(answer)

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

        if answer_scores:
            best_idx = max(range(len(candidate_answers)), key=lambda i: answer_scores[i])
            final_answer = candidate_answers[best_idx]
        else:
            final_answer = candidate_answers[0] if candidate_answers else ""

        print(f"Question: {question}")
        print("Queries used:")
        for q in queries:
            print(f"- {q}")
        print(f"Original Answer: {gold}")
        print("Candidate answers and confidences:")
        for i, (ans, conf) in enumerate(zip(candidate_answers, answer_scores)):
            print(f"  [{i}] {ans!r} (conf={conf:.2f})")
        print("Chosen Contexts:")
        for ctx in selected_ctxs:
            print(f"- {ctx[:100]}...")
        print(f"Chosen final answer: {final_answer}")
        print("-" * 60)


if __name__ == "__main__":
    run_simple_pipeline("hotpot_dev_distractor_v1.json", n_samples=10)
