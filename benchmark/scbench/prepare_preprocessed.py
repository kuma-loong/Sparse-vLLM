#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(THIS_DIR))

from datasets import Dataset, load_dataset

from benchmark.common.paths import scbench_preprocessed_root
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, create_scdq_prompt


DEFAULT_TASKS = [
    "scbench_kv",
    "scbench_qa_eng",
    "scbench_summary_with_needles",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare raw SCBench snapshots for run_scbench_preprocessed.py."
    )
    parser.add_argument("--source_root", required=True, help="Raw SCBench root from microsoft/SCBench.")
    parser.add_argument(
        "--output_root",
        default=None,
        help=(
            "Directory for flat scbench_*.parquet files. Defaults to "
            "SVLLM_SCBENCH_PREPROCESSED_ROOT/SCBENCH_PREPROCESSED_ROOT when set, "
            "otherwise <source_root>-preprocessed."
        ),
    )
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated SCBench tasks to prepare, or 'all' for all local tasks.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(args.output_root).expanduser()
    env_root = scbench_preprocessed_root()
    if env_root is not None:
        return env_root
    return Path(f"{Path(args.source_root).expanduser()}-preprocessed")


def discover_tasks(source_root: Path) -> list[str]:
    tasks = set()
    for task_dir in source_root.glob("scbench_*"):
        if task_dir.is_dir() and any(task_dir.glob("*.parquet")):
            tasks.add(task_dir.name)
    data_dir = source_root / "data"
    if data_dir.is_dir():
        tasks.update(path.stem for path in data_dir.glob("scbench_*.jsonl"))
    return sorted(task for task in tasks if task in DATA_NAME_TO_MAX_NEW_TOKENS)


def load_raw_scbench(source_root: Path, task: str):
    task_dir = source_root / task
    parquet_files = sorted(str(path) for path in task_dir.glob("*.parquet"))
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, split="train")

    jsonl_path = source_root / "data" / f"{task}.jsonl"
    if jsonl_path.is_file():
        return load_dataset("json", data_files=str(jsonl_path), split="train")

    raise FileNotFoundError(
        f"Missing raw SCBench task data for {task}. Expected {task_dir}/*.parquet "
        f"or {jsonl_path}."
    )


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if isinstance(normalized.get("multi_turns"), str):
        normalized["multi_turns"] = json.loads(normalized["multi_turns"])
    if not isinstance(normalized.get("multi_turns"), list):
        raise ValueError("SCBench row is missing list-valued multi_turns.")
    return normalized


def add_metadata(output: dict[str, Any], row: dict[str, Any], idx: int, task: str) -> dict[str, Any]:
    output = dict(output)
    output["id"] = row.get("id", idx)

    for key in ["lang", "repo"]:
        if key in row:
            output[key] = row[key]

    if "repoqa" in task:
        output["func_name"] = [turn.get("name") for turn in row["multi_turns"]]

    return output


def validate_preprocessed_row(row: dict[str, Any], task: str, idx: int) -> None:
    prompts = row.get("prompts")
    ground_truth = row.get("ground_truth")
    if not isinstance(prompts, list) or len(prompts) < 2:
        raise ValueError(f"{task}[{idx}] has invalid prompts.")
    if not isinstance(ground_truth, list) or len(ground_truth) != len(prompts) - 1:
        raise ValueError(f"{task}[{idx}] has invalid ground_truth length.")
    if "repoqa" in task:
        func_name = row.get("func_name")
        if not isinstance(func_name, list) or len(func_name) != len(ground_truth):
            raise ValueError(f"{task}[{idx}] has invalid func_name metadata.")


def prepare_task(source_root: Path, output_root: Path, task: str, overwrite: bool) -> dict[str, Any]:
    output_path = output_root / f"{task}.parquet"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    dataset = load_raw_scbench(source_root, task)
    rows = []
    for idx, raw_row in enumerate(dataset):
        row = normalize_row(dict(raw_row))
        converted = create_scdq_prompt(row, task, tok=None, use_chat_template=False)
        if converted is None:
            raise ValueError(f"Unsupported SCBench task for SCDQ preprocessing: {task}")
        converted = add_metadata(converted, row, idx, task)
        validate_preprocessed_row(converted, task, idx)
        rows.append(converted)

    output_root.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))
    return {
        "task": task,
        "rows": len(rows),
        "output_path": str(output_path),
        "columns": sorted(rows[0].keys()) if rows else [],
    }


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"SCBench source root does not exist: {source_root}")

    output_root = resolve_output_root(args)
    tasks = discover_tasks(source_root) if args.tasks == "all" else split_csv(args.tasks)
    if not tasks:
        raise ValueError("No SCBench tasks selected.")

    unknown = sorted(task for task in tasks if task not in DATA_NAME_TO_MAX_NEW_TOKENS)
    if unknown:
        raise ValueError(f"Unknown SCBench tasks: {unknown}")

    summaries = [prepare_task(source_root, output_root, task, args.overwrite) for task in tasks]
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "tasks": summaries,
    }
    summary_path = output_root / "prepare_scbench_preprocessed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
