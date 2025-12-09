from typing import List, Dict
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

def load_hotpot_json(path: str | Path) -> List[Dict]:
    """
    Load the HotpotQA dataset JSON.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_context_paragraphs(sample: Dict, include_titles: bool = True) -> List[str]:
    """
    Turn the 'context' field of one HotpotQA sample into a list of paragraphs.
    If include_titles=True, prefix each paragraph with its title.
    """
    contexts: List[str] = []
    for title, sentences in sample["context"]:
        paragraph = " ".join(sentences)
        if include_titles:
            contexts.append(f"{title}: {paragraph}")
        else:
            contexts.append(paragraph)
    return contexts

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