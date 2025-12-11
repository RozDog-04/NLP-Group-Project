# Multi-Trajectory RAG on HotpotQA

This repository contains our coursework implementation of a **multi-trajectory Retrieval-Augmented Generation (RAG)** system on the **HotpotQA** dev-distractor split. The core idea is:

- Use a **BM25S** index over Wikipedia chunks,
- Generate **multiple query reformulations** per question (multi-trajectory),
- Retrieve and re-rank evidence with an **LLM-based reranker**,
- Generate **short Hotpot-style answers** with an LLM and **confidence-based voting**.

The entire environment setup, data download, indexing, and prediction pipeline can be run via a single script: `deploy.sh`.

---

## 1. Prerequisites

Before running `deploy.sh`, you need:

- **Operating system**: Linux or macOS (or WSL on Windows). The script uses Bash, `git`, `wget`, and `unzip`.
- **Python**: Python 3.8+ available as `python3` (or set `PYTHON` env var to your Python executable).
- **Pip**: A working `pip` (the script will upgrade it).
- **Git**: To clone the data repository containing pre-processed splits.
- **Internet access**: To download:
  - HotpotQA dev-distractor JSON,
  - GloVe embeddings,
  - spaCy English model,
  - BM25 chunks from a separate data repo.

> **API keys**  
> If your code uses external LLM APIs (e.g. Mistral), you should provide your key via environment variable (e.g. `MISTRAL_API_KEY`) or a `.env` file. The deployment script itself does **not** set this.

---

## 2. Quick Start: One-Command Deployment

From the project root (where `deploy.sh` and `requirements.txt` live), run:

```bash
chmod +x deploy.sh
./deploy.sh




