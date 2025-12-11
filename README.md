# Multi‑Trajectory RAG for HotpotQA

This project answers HotpotQA questions using multiple query reformulations, BM25 retrieval, optional LLM reranking, and answer voting with confidence/consensus.

---
## Prerequisites
- Python 3.8+
- Bash + basic CLI tools (`git`, `wget`, `unzip`)
- Internet access to download data/models
- LLM key: set `MISTRAL_API_KEY` (env or `.env`)
- Windows: use WSL for best results; some scripts emit harmless “resource module not available on Windows”.

---
## Setup
1) Install deps:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

2) Data & BM25 index:
- Ensure Hotpot dev JSON is present (e.g., `hotpot_dev_distractor_v1.json`).
- Ensure BM25 index/store exist at `data/index/bm25s_index` and `data/index/bm25_store.pkl`.
- If missing, build from chunks:
```bash
python build_BM25_index.py \
  --chunks data/processed/chunks.jsonl \
  --out-index data/index/bm25s_index \
  --out-store data/index/bm25s_store.pkl
```
Create `data/processed/chunks.jsonl` with your chunker pipeline first.

3) API key: set `MISTRAL_API_KEY` in your shell or `.env`.

---
## Running

### Sample run (console)
Multi‑trajectory demo over a small subset:
```bash
python predict_sample.py  # adjust n_samples/top_k_for_answer inside the file
```
Prints queries per trajectory, chosen contexts, candidate answers, confidences, and final answer.

### Full dev to JSON
Multi‑trajectory + voting over the full dev set:
```bash
python predict_full.py \
  --dev_json hotpot_dev_distractor_v1.json \
  --output predictions_dev.json
```
Edit defaults in the file for `top_k_for_answer` or dataset paths if needed.

---
## How it works (core pieces)
- `multi_BM25_retrieval.py` + `question_reformulating.py`: build multiple query trajectories (original/rewrite/decomp/entity) and retrieve BM25 contexts.
- `llm_pipeline.py`: Mistral-based AnswerGenerator, ContextReranker (optional), query rewriting helpers, and answer confidence scoring.
- `predict_sample.py`: console runner with multi-trajectory retrieval, rerank, consensus (frequency + confidence), and debug prints.
- `predict_full.py`: same logic as `predict_sample` but runs over the whole dev set and writes `predictions_dev.json`.
- `build_BM25_index.py`: build BM25 index/store from chunked text.

---
## Tips
- Keep generated indices/chunks out of git (`data/index/`, `data/processed/chunks*.jsonl` in `.gitignore`).
- Confidence scoring encourages graded values; consensus prefers overlapping, non-fallback answers.
- Adjust `top_k_for_answer` to change how many reranked contexts feed the answer model (default 10 in full run).
