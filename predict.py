import numpy as np
from sentence_transformers import SentenceTransformer

from data_utils import load_hotpot_json, extract_context_paragraphs
from llm_pipeline import AnswerGenerator, ContextReranker
from BM25S_retrieval import BM25Retriever

# class DenseRetriever:
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.encoder = SentenceTransformer(model_name)
# 
#     def embed(self, texts):
#         return self.encoder.encode(
#             list(texts),
#             convert_to_numpy=True,
#             normalize_embeddings=True,
#             show_progress_bar=False,
#         )
# 
#     def retrieve(self, query: str, contexts, top_k: int = 10):
#         if not contexts:
#             return []
#         ctx_embs = self.embed(contexts)
#         q_emb = self.embed([query])[0]
#         scores = ctx_embs @ q_emb
#         indices = np.argsort(scores)[::-1][:top_k]
#         return indices.tolist(), scores


def run_simple_pipeline(dev_json_path: str, n_samples: int = 3, top_k_for_answer: int = 5):
    data = load_hotpot_json(dev_json_path)
    
    # Initializes BM25S Retriever
    # Can adjust paths as needed or pass them in
    index_path = "data/index/bm25s_index"
    store_path = "data/index/bm25_store.pkl"
    retriever = BM25Retriever(index_path=index_path, store_path=store_path)
    
    answer_gen = AnswerGenerator()
    reranker = ContextReranker()

    for sample in data[:n_samples]:
        question = sample.get("question", "")
        gold = sample.get("answer", "")

        # Retrieves from BM25
        results, scores = retriever.retrieve(question, top_k=10)
        
        # Extracts text from results for reranking/answering
        retrieved_ctxs = [r["text"] for r in results]
        
        if reranker is not None:
            rerank_scores = reranker.score(question, retrieved_ctxs)
            reranked_local = sorted(range(len(retrieved_ctxs)), key=lambda i: rerank_scores[i], reverse=True)
            ranked_ctxs = [retrieved_ctxs[i] for i in reranked_local]
            ranked_scores = [rerank_scores[i] for i in reranked_local]
        else:
            ranked_ctxs = retrieved_ctxs
            ranked_scores = scores

        top_k = min(top_k_for_answer, len(ranked_ctxs))
        selected_ctxs = ranked_ctxs[:top_k]
        top_scores = ranked_scores[:top_k]

        answer = answer_gen.generate_answer(question, selected_ctxs)

        print(f"Question: {question}")
        print(f"Original Answer: {gold}")
        print("Chosen Contexts (with confidence):")
        for i, ctx in enumerate(selected_ctxs):
             print(f"- conf={top_scores[i]:.2f} :: {ctx[:100]}...")
        print(f"Generated Answer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    run_simple_pipeline("hotpot_dev_distractor_v1.json", n_samples=10)
