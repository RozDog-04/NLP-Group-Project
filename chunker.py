import os
import bz2
from typing import Iterator, Dict, Any, List, Tuple
import json
import argparse
import re

"""
To run:
python chunker.py \                             
  --abstracts-dir data/wiki_abstracts/enwiki-20171001-pages-meta-current-withlinks-abstracts \
  --out data/processed/chunks.jsonl \
  --chunk-size 200 \
  --overlap 50

To read the jsonl output file:
head -n 10 data/processed/chunks.jsonl
"""
YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")

def get_years(text: str):
    years = YEAR_RE.findall(text)
    years = sorted({int(x) for x in years})
    return years

def iter_wiki_json_objects(abstracts_dir: str) -> Iterator[Dict[str, Any]]:
    """
    Iterates over all JSON objects in all .bz2 files under abstracts_dir
    Each line in the inner .bz2 files is expected to be a JSON object
    """
    for root, _, files in os.walk(abstracts_dir):
        for filename in sorted(files):
            if not filename.endswith(".bz2"):
                continue
            path = os.path.join(root, filename)
            print(f"[INFO] Reading {path}")
            with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue
                    yield obj


def extract_introduction_text(obj: Dict[str, Any]) -> str:
    """
    Extracts the abstract (intro) text from the wiki JSON object

    The format described by official HotpotQA is:
    {
    "id":
    "url":
    "title":
    "text":
    "charoffset":
    }
    
    We handle:
    - text as a list of paragraphs
    - text as a list of strings
    - text as a plain string
    """
    text = obj.get("text")

    if isinstance(text, str):
        return text.strip()

    if isinstance(text, list):
        paragraphs: List[str] = []
        for para in text:
            if isinstance(para, list):
                paragraphs.append(" ".join(para))
            elif isinstance(para, str):
                paragraphs.append(para)
            # else ignores any other weird options
        full = "\n\n".join(paragraphs)
        return full.strip()

    # Fallback for when ther is no usable text
    return ""


def sliding_window_chunks(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> List[Tuple[str, int, int]]:
    """
    Word level sliding window chunking

    Returns a list of (chunk_text, start_word_idx, end_word_idx)
    """
    words = text.split()
    if not words:
        return []

    n = len(words)
    chunks: List[Tuple[str, int, int]] = []
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, start, end))

        if end == n:
            break

        # Moves window with the overlap
        start = max(0, end - overlap)

    return chunks

def build_chunks(
    abstracts_dir: str,
    out_path: str,
    chunk_size: int = 200,
    overlap: int = 50,
    max_pages: int = None,
) -> None:
    """
    Main routine that reads wiki abstracts, chunks them, and writes to data/processed/chunks.jsonl
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    num_pages = 0
    num_chunks = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for obj in iter_wiki_json_objects(abstracts_dir):
            # Stops early if max_pages is set
            # Only used max_pages for testing 
            if max_pages is not None and num_pages >= max_pages:
                break

            page_id = obj.get("id")
            title = obj.get("title")
            url = obj.get("url")

            intro = extract_introduction_text(obj)
            if not intro:
                continue

            chunks = sliding_window_chunks(intro, chunk_size=chunk_size, overlap=overlap)
            if not chunks:
                continue

            for idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
                chunk_id = f"{page_id}_{idx}"
                
                years = get_years(chunk_text)

                record = {
                    "chunk_id": str(chunk_id),
                    "doc_id": page_id,
                    "title": title,
                    "url": url,
                    "text": chunk_text,
                    "start_word": start_idx,
                    "end_word": end_idx,
                    "metadata": {
                        "years": years,
                        "has_year": bool(years),
                    },
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_chunks += 1

            num_pages += 1
            if num_pages % 1000 == 0:
                print(f"[INFO] Processed {num_pages} pages, {num_chunks} chunks have been made")

    print(f"[DONE] Finished -> Pages processed: {num_pages}, chunks written: {num_chunks}")
    print(f"[DONE] Output file: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Builds chunked wiki abstracts JSONL")
    parser.add_argument(
        "--abstracts-dir",
        type=str,
        required=True,
        help="Directory with inner .bz2 wiki abstract files"
             "(e.g. data/raw_data/enwiki-20171001-pages-meta-current-withlinks-abstracts)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/chunks.jsonl",
        help="Output jsonl path for chunks",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Number of words per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Number of overlapping words between consecutive chunks",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Mainly for testing: limit number of pages (e.g. 2000)",
    )

    args = parser.parse_args()

    build_chunks(
        abstracts_dir=args.abstracts_dir,
        out_path=args.out,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
