import argparse
import pickle
from typing import List, Dict, Any, Tuple
import bm25s

"""
To test:
python BM25S_retrieval.py \
  --index data/index/bm25s_index \
  --store data/index/bm25_store.pkl \
  --query "What is machine learning?" \
  --top-k 10
"""

class BM25Retriever:
    def __init__(self, index_path: str, store_path: str):
        print(f"[INFO] Loading BM25 index from {index_path}...")
        self.retriever = bm25s.BM25.load(index_path, load_corpus=False)
        
        print(f"[INFO] Loading BM25 store from {store_path}...")
        with open(store_path, "rb") as f:
            self.store = pickle.load(f)
            
        self.texts: List[str] = self.store["texts"]
        self.meta: List[Dict[str, Any]] = self.store["meta"]

    def retrieve(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        # Tokenizes query using the same tokenizer used for indexing
        query_tokens = bm25s.tokenize(query)

        # Gets doc IDs + scores
        doc_ids, scores = self.retriever.retrieve(query_tokens, k=top_k)
        
        # bm25s returns shape (n_queries, k), we only have 1 query
        doc_ids = doc_ids[0]
        scores = scores[0]

        results: List[Dict[str, Any]] = []
        result_scores: List[float] = []
        
        for doc_idx, score in zip(doc_ids, scores):
            i = int(doc_idx)   # numeric ID
            results.append(
                {
                    "score": float(score),
                    "text": self.texts[i],
                    "meta": self.meta[i],
                }
            )
            result_scores.append(float(score))

        return results, result_scores

def main():
    parser = argparse.ArgumentParser(description="Searches the BM25S index")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--store", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    print(f"[INFO] Searching for query: {args.query!r}")
    
    # Initialize retriever
    retriever = BM25Retriever(args.index, args.store)
    
    # Retrieve
    results, _ = retriever.retrieve(args.query, top_k=args.top_k)

    for r in results:
        print("=" * 80)
        print(f"Score: {r['score']:.4f}")
        print("Title:", r["meta"].get("title"))
        print("URL  :", r["meta"].get("url"))
        print("Years:", r["meta"].get("metadata", {}).get("years"))
        print("--- Text snippet ---")
        print(r["text"][:400], "...")

if __name__ == "__main__":
    main()
