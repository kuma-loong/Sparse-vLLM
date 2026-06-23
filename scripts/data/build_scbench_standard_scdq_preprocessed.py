from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, load_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
SCBENCH_DIR = REPO_ROOT / "benchmark" / "scbench"
sys.path.append(str(SCBENCH_DIR))

from eval_utils import create_scdq_prompt  # noqa: E402


def parse_args():
    parser = ArgumentParser(
        description=(
            "Build KVzip-compatible SCDQ parquet files directly from the "
            "standard microsoft/SCBench split."
        )
    )
    parser.add_argument("--task", required=True, help="SCBench config name, e.g. scbench_qa_eng")
    parser.add_argument(
        "--output-root",
        required=True,
        help="Directory where <task>.parquet and a manifest JSON will be written.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", default="microsoft/SCBench")
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Optional first-N limit for smoke files. Default writes the full split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    source = load_dataset(args.dataset, args.task, split=args.split)
    if args.limit >= 0:
        source = source.select(range(min(args.limit, len(source))))

    rows = []
    for idx, row in enumerate(source):
        encoded = create_scdq_prompt(
            row,
            data_name=args.task,
            tok=None,
            use_chat_template=False,
        )
        out_row = {
            "prompts": encoded["prompts"],
            "ground_truth": encoded["ground_truth"],
        }
        if "task" in encoded:
            out_row["task"] = encoded["task"]
        rows.append(out_row)

        if len(out_row["prompts"]) != len(out_row["ground_truth"]) + 1:
            raise ValueError(
                f"Invalid SCDQ row {idx}: prompts={len(out_row['prompts'])}, "
                f"ground_truth={len(out_row['ground_truth'])}"
            )

    if not rows:
        raise ValueError(f"No rows generated for {args.dataset}/{args.task} split={args.split}")

    parquet_path = output_root / f"{args.task}.parquet"
    Dataset.from_list(rows).to_parquet(str(parquet_path))

    manifest = {
        "dataset": args.dataset,
        "task": args.task,
        "split": args.split,
        "rows": len(rows),
        "format": "KVzip-compatible SCDQ parquet",
        "columns": list(rows[0].keys()),
        "parquet": parquet_path.name,
        "source_builder": str(Path(__file__).relative_to(REPO_ROOT)),
    }
    manifest_path = output_root / f"{args.task}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
