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

        # Processes each trajectory
        for traj_name, info in traj_results.items():
            queries = info["queries"]
            docs = info["docs"]
            
            # Extract text
            retrieved_ctxs = [d["text"] for d in docs]
            
            if not retrieved_ctxs:
                print(f"Trajectory: {traj_name} (No docs found)")
                continue

            # Reranks
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

            print(f"Trajectory: {traj_name}")
            print(f"  Queries: {queries}")
            print(f"  Generated Answer: {answer}")
            
        print("=" * 60)


if __name__ == "__main__":
    run_simple_pipeline("hotpot_dev_distractor_v1.json", n_samples=5)
