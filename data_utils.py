from typing import List, Dict
import json
from pathlib import Path


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
