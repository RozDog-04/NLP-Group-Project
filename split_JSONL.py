import argparse
import os

"""
To run:
python split_jsonl.py \
  --input data/processed/chunks.jsonl \
  --out-dir data/processed/splits \
  --max-mb 100 \
  --prefix chunks_part
"""

DEFAULT_MAX_MB = 100
DEFAULT_PREFIX = "chunks_part"

def split_jsonl(
    input_path: str,
    out_dir: str,
    max_mb: int = DEFAULT_MAX_MB,
    prefix: str = DEFAULT_PREFIX,
) -> None:
    """
    Splits a large JSONL file into multiple smaller JSONL files
    no bigger than 100MB
    """
    os.makedirs(out_dir, exist_ok=True)

    max_bytes = max_mb * 1024 * 1024  # converts MB to bytes

    part_idx = 1
    current_bytes = 0
    out_f = None

    def open_new_file(idx: int):
        filename = f"{prefix}_{idx:05d}.jsonl"
        path = os.path.join(out_dir, filename)
        print(f"[INFO] Opening new file: {path}")
        return open(path, "w", encoding="utf-8"), path

    with open(input_path, "r", encoding="utf-8") as in_f:
        out_f, current_path = open_new_file(part_idx)

        for line_num, line in enumerate(in_f, start=1):
            # Keeps the line exactly the same as the original
            if not line:
                continue

            line_bytes = len(line.encode("utf-8"))

            # If this line would push us over the limit it starts a new file
            if current_bytes + line_bytes > max_bytes and current_bytes > 0:
                out_f.close()
                part_idx += 1
                current_bytes = 0
                out_f, current_path = open_new_file(part_idx)

            out_f.write(line)
            current_bytes += line_bytes

        out_f.close()

    print(f"[DONE] Finished splitting {input_path} into {part_idx} files in '{out_dir}'.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB)
    parser.add_argument("--prefix", type=str, default=DEFAULT_PREFIX)

    args = parser.parse_args()

    split_jsonl(
        input_path=args.input,
        out_dir=args.out_dir,
        max_mb=args.max_mb,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
