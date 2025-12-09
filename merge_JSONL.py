import os
import argparse
import glob

"""
To run:
python merge_jsonl.py \
  --in-dir data/processed/splits \
  --out data/processed/chunks.jsonl \
  --prefix chunks_part
"""

def merge_jsonl_parts(
    in_dir: str,
    out_path: str,
    prefix: str = "chunks_part",
) -> None:
    """
    Merges multiple JSONL part files into a single JSONL file
    Should be identical to the original JSONL chunks file
    """

    pattern = os.path.join(in_dir, f"{prefix}_*.jsonl")
    part_files = sorted(glob.glob(pattern))

    if not part_files:
        print(f"[ERROR] No files matching {pattern}")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"[INFO] Found {len(part_files)} part file(s).")
    print(f"[INFO] Writing merged JSONL out to {out_path}")

    total_lines = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for part in part_files:
            print(f"[INFO] Merging {part}")
            with open(part, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_lines += 1

    print(f"[DONE] Merged {len(part_files)} files into {out_path}")
    print(f"[DONE] Total lines written: {total_lines}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="chunks_part")

    args = parser.parse_args()

    merge_jsonl_parts(
        in_dir=args.in_dir,
        out_path=args.out,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
