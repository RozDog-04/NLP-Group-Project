"""
Quick helper script to print the first three entries from a JSON file.

Usage:
    python preview_json.py --path hotpot_dev_distractor_v1.json
    python preview_json.py --path hotpot_dev_distractor_v1.json --count 5
"""

import argparse
import json
from pathlib import Path
from typing import Any, List


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the first few entries from a JSON file.")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("hotpot_dev_distractor_v1.json"),
        help="Path to the JSON file (default: hotpot_dev_distractor_v1.json).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="How many entries to show from the start of the file (default: 3).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("preview_output.json"),
        help="File to write the preview JSON into (default: preview_output.json).",
    )
    args = parser.parse_args()

    data = load_json(args.path)

    if isinstance(data, list):
        subset: List[Any] = []
        for entry in data[: args.count]:
            if isinstance(entry, dict):
                # add an explicit "query" field mirroring "question" for clarity
                entry = {**entry}
                if "question" in entry and "query" not in entry:
                    entry["query"] = entry["question"]
            subset.append(entry)
    else:
        # If the JSON is not a list, just print the whole object.
        subset = data

    with args.out.open("w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
