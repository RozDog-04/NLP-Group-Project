import argparse
import pickle
import bm25s
from data_utils import load_chunks_jsonl

"""
To run:
python build_BM25_index.py \
  --chunks data/processed/chunks.jsonl \
  --out-index data/index/bm25s_index \
  --out-store data/index/bm25_store.pkl
"""

def build_bm25s_index(
    chunks_path: str,
    out_index_path: str,
    out_store_path: str,
    max_docs: int | None = None,
) -> None:
    print(f"[INFO] Loading chunks from {chunks_path}")
    texts, meta = load_chunks_jsonl(chunks_path, max_docs=max_docs)
    print(f"[INFO] Loaded {len(texts)} chunks")

    if not texts:
        print("[WARNING] No text loaded. Aborting process")
        return

    # Tokenize with BM25S tokenizer (handles stopwords etc.)
    print("[INFO] Tokenizing corpus with bm25s.tokenize()")
    corpus_tokens = bm25s.tokenize(texts, stopwords="en")

    print("[INFO] Building BM25S index (Using Lucene style BM25)")
    retriever = bm25s.BM25(corpus=texts, method="lucene")
    retriever.index(corpus_tokens)
    print("[INFO] BM25S index built")

    print(f"[INFO] Saving BM25S index to {out_index_path}")
    retriever.save(out_index_path)  # Uses BM25S's own save formatting

    print(f"[INFO] Saving texts & meta store to {out_store_path}")
    with open(out_store_path, "wb") as f:
        pickle.dump({"texts": texts, "meta": meta}, f)

    print("[DONE] BM25S index + store saved")


def main():
    parser = argparse.ArgumentParser(description="Builds BM25S index over wiki chunks")
    parser.add_argument("--chunks", type=str, required=True)
    parser.add_argument("--out-index", type=str, default="data/index/bm25s_index")
    parser.add_argument("--out-store", type=str, default="data/index/bm25_store.pkl")
    parser.add_argument("--max-docs", type=int, default=None)

    args = parser.parse_args()

    build_bm25s_index(
        chunks_path=args.chunks,
        out_index_path=args.out_index,
        out_store_path=args.out_store,
        max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
