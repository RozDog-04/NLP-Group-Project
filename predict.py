import numpy as np
from sentence_transformers import SentenceTransformer

from data_utils import load_hotpot_json, extract_context_paragraphs
from llm_pipeline import AnswerGenerator, ContextReranker


class DenseRetriever:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.encoder = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.encoder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, contexts, top_k: int = 10):
        if not contexts:
            return []
        ctx_embs = self.embed(contexts)
        q_emb = self.embed([query])[0]
        scores = ctx_embs @ q_emb
        indices = np.argsort(scores)[::-1][:top_k]
        return indices.tolist(), scores


def run_simple_pipeline(dev_json_path: str, n_samples: int = 3, top_k_for_answer: int = 5):
    data = load_hotpot_json(dev_json_path)
    retriever = DenseRetriever()
    answer_gen = AnswerGenerator()
    reranker = ContextReranker()

    for sample in data[:n_samples]:
        question = sample.get("question", "")
        gold = sample.get("answer", "")

        contexts = extract_context_paragraphs(sample, include_titles=True)

        dense_indices, dense_scores = retriever.retrieve(question, contexts, top_k=len(contexts))

        if reranker is not None:
            dense_ctxs = [contexts[i] for i in dense_indices]
            rerank_scores = reranker.score(question, dense_ctxs)
            reranked_local = sorted(range(len(dense_ctxs)), key=lambda i: rerank_scores[i], reverse=True)
            ranked_indices = [dense_indices[i] for i in reranked_local]
            ranked_scores = [rerank_scores[i] for i in reranked_local]
        else:
            ranked_indices = dense_indices
            ranked_scores = [float(dense_scores[i]) for i in ranked_indices]

        top_k = min(top_k_for_answer, len(ranked_indices))
        top_indices = ranked_indices[:top_k]
        top_scores = ranked_scores[:top_k]
        selected_ctxs = [contexts[i] for i in top_indices]

        answer = answer_gen.generate_answer(question, selected_ctxs)

        print(f"Question: {question}")
        print(f"Original Answer: {gold}")
        print("Chosen Contexts (with confidence):")
        for idx, conf in zip(top_indices, top_scores):
            print(f"- idx={idx} conf={conf:.2f} :: {contexts[idx]}")
        print(f"Generated Answer: {answer}")
        print("-" * 60)


if __name__ == "__main__":
    run_simple_pipeline("hotpot_dev_distractor_v1.json", n_samples=3)
