import argparse
import pickle
from typing import List, Dict, Any
import bm25s

"""
To test:
python BM25S_retrieval.py \
  --index data/index/bm25s_index \
  --store data/index/bm25_store.pkl \
  --query "What is machine learning?" \
  --top-k 10
"""

def bm25s_search(
    index_path: str,
    store_path: str,
    query: str,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    # Loads retriever without loading a corpus
    retriever = bm25s.BM25.load(index_path, load_corpus=False)

    # Loads store (has the texts + metadata)
    with open(store_path, "rb") as f:
        store = pickle.load(f)

    texts: List[str] = store["texts"]
    meta: List[Dict[str, Any]] = store["meta"]

    # Tokenizes query using the same tokenizer used for indexing
    query_tokens = bm25s.tokenize(query)

    # Gets doc IDs + scores
    doc_ids, scores = retriever.retrieve(query_tokens, k=top_k)
    doc_ids = doc_ids[0]
    scores = scores[0]

    results: List[Dict[str, Any]] = []
    for doc_idx, score in zip(doc_ids, scores):
        i = int(doc_idx)   # numeric ID
        results.append(
            {
                "score": float(score),
                "text": texts[i],
                "meta": meta[i],
            }
        )

    return results

def main():
    parser = argparse.ArgumentParser(description="Searches the BM25S index")
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--store", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()

    print(f"[INFO] Searching for query: {args.query!r}")
    results = bm25s_search(args.index, args.store, args.query, top_k=args.top_k)

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
