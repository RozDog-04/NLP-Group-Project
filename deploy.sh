#!/usr/bin/env bash
set -euo pipefail

########################################
# CONFIG
########################################

# GitHub repo that contains data/processed/splits
DATA_REPO_URL="https://github.com/RozDog-04/NLP-Group-Project.git"
DATA_REPO_BRANCH="main"
DATA_REPO_TMP_DIR=".tmp_data_repo"

# Python executable (change to python if needed)
PYTHON=${PYTHON:-python3}

########################################
# 1. Install Python dependencies
########################################

echo "[1/7] Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

########################################
# 2. Download data/processed/splits from GitHub
########################################

echo "[2/7] Downloading data/processed/splits from GitHub..."

rm -rf "$DATA_REPO_TMP_DIR"
git clone --depth 1 --branch "$DATA_REPO_BRANCH" "$DATA_REPO_URL" "$DATA_REPO_TMP_DIR"

mkdir -p data/processed
rm -rf data/processed/splits
cp -r "$DATA_REPO_TMP_DIR/data/processed/splits" data/processed/

rm -rf "$DATA_REPO_TMP_DIR"

########################################
# 3. Download Hotpot dev-distractor + GloVe + SpaCy model
########################################

echo "[3/7] Downloading Hotpot dev-distractor dataset into project root..."
# This will end up in the same directory as deploy.sh
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
  -O hotpot_dev_distractor_v1.json

echo "Downloading GloVe embeddings into project root..."
GLOVE_DIR=./
mkdir -p "$GLOVE_DIR"
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O "$GLOVE_DIR/glove.840B.300d.zip"
unzip -o "$GLOVE_DIR/glove.840B.300d.zip" -d "$GLOVE_DIR"

echo "Downloading SpaCy English model..."
$PYTHON -m spacy download en

########################################
# 4. Merge JSONL chunks
########################################

echo "[4/7] Running merge_jsonl.py..."
$PYTHON merge_jsonl.py \
  --in-dir data/processed/splits \
  --out data/processed/chunks.jsonl \
  --prefix chunks_part

########################################
# 5. Build BM25 index
########################################

echo "[5/7] Building BM25 index..."
mkdir -p data/index
$PYTHON build_BM25_index.py \
  --chunks data/processed/chunks.jsonl \
  --out-index data/index/bm25s_index \
  --out-store data/index/bm25_store.pkl

########################################
# 6. Run predict_sample.py
########################################

echo "[6/7] Running predict_sample.py..."
$PYTHON predict_sample.py

echo "deploy.sh appears to have succeeded. Running predict_full.py in 5 seconds..."

########################################
# 7. Wait a bit, then run predict_full.py
########################################

sleep 5

echo "[7/7] Running predict_full.py..."
$PYTHON predict_full.py

echo "deploy.sh finished successfully."
