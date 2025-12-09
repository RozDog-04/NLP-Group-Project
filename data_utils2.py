import json
from typing import List, Dict, Any, Tuple, Optional

def load_chunks_jsonl(
    path: str,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Loads chunks from the JSONL file

    Returns the:
        texts: raw text per chunk
        meta: metadata dict per chunk (chunk_id, doc_id, title, url, metadata)
    """
    texts: List[str] = []
    meta: List[Dict[str, Any]] = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break

            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue

            texts.append(text)
            meta.append(
                {
                    "chunk_id": obj.get("chunk_id"),
                    "doc_id": obj.get("doc_id"),
                    "title": obj.get("title"),
                    "url": obj.get("url"),
                    "metadata": obj.get("metadata", {}),
                }
            )

    return texts, meta
