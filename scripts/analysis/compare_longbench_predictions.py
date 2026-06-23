#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
LONGBENCH_DIR = REPO_ROOT / "benchmark" / "long_bench"
sys.path.insert(0, str(LONGBENCH_DIR))

from eval import TASK_HIERARCHY, aggregate_category_scores, dataset2metric  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two LongBench prediction directories with task and sample-level deltas."
    )
    parser.add_argument("--ref", required=True, type=Path, help="Reference prediction directory, usually HF.")
    parser.add_argument("--test", required=True, type=Path, help="Test prediction directory, usually Sparse-VLLM.")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON report path.")
    parser.add_argument("--top_n", default=20, type=int, help="Number of largest per-sample deltas to keep per task.")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Allow comparing only the shared prefix when task row counts differ.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def _score_row(task: str, row: dict[str, Any]) -> float:
    if "pred" not in row or "answers" not in row:
        raise ValueError(f"Task {task}: row missing required pred/answers fields.")
    all_classes = row.get("all_classes")
    pred = row["pred"]
    if task in ["trec", "triviaqa", "samsum", "lsht"]:
        pred = pred.lstrip("\n").split("\n")[0]
    score = 0.0
    for answer in row["answers"]:
        score = max(score, float(dataset2metric[task](pred, answer, all_classes=all_classes)))
    return score


def _task_files(ref_dir: Path, test_dir: Path) -> list[str]:
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference directory does not exist: {ref_dir}")
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir}")
    ref_tasks = {p.stem for p in ref_dir.glob("*.jsonl")}
    test_tasks = {p.stem for p in test_dir.glob("*.jsonl")}
    common = sorted(ref_tasks & test_tasks)
    if not common:
        raise ValueError(f"No common .jsonl task files between {ref_dir} and {test_dir}.")
    return common


def _round(value: float) -> float:
    return round(float(value), 4)


def compare_task(task: str, ref_dir: Path, test_dir: Path, *, top_n: int, allow_partial: bool) -> dict[str, Any]:
    ref_rows = _read_jsonl(ref_dir / f"{task}.jsonl")
    test_rows = _read_jsonl(test_dir / f"{task}.jsonl")
    if len(ref_rows) != len(test_rows):
        if not allow_partial:
            raise ValueError(
                f"Row count mismatch for {task}: ref={len(ref_rows)} test={len(test_rows)}. "
                "Pass --allow_partial to compare the shared prefix explicitly."
            )
        n = min(len(ref_rows), len(test_rows))
        ref_rows = ref_rows[:n]
        test_rows = test_rows[:n]
    if not ref_rows:
        raise ValueError(f"Task {task} has no rows to compare.")

    ref_scores = [_score_row(task, row) for row in ref_rows]
    test_scores = [_score_row(task, row) for row in test_rows]
    deltas = [test - ref for ref, test in zip(ref_scores, test_scores)]
    samples = []
    for idx, (ref_row, test_row, ref_score, test_score, delta) in enumerate(
        zip(ref_rows, test_rows, ref_scores, test_scores, deltas)
    ):
        samples.append(
            {
                "idx": idx,
                "ref_score": _round(ref_score * 100.0),
                "test_score": _round(test_score * 100.0),
                "delta": _round(delta * 100.0),
                "answers": ref_row.get("answers"),
                "ref_pred": ref_row.get("pred", ""),
                "test_pred": test_row.get("pred", ""),
                "length": ref_row.get("length"),
            }
        )

    worst = sorted(samples, key=lambda item: item["delta"])[:top_n]
    best = sorted(samples, key=lambda item: item["delta"], reverse=True)[:top_n]
    changed = sum(1 for ref_row, test_row in zip(ref_rows, test_rows) if ref_row.get("pred") != test_row.get("pred"))
    return {
        "task": task,
        "n": len(ref_rows),
        "ref_score": round(100.0 * float(np.mean(ref_scores)), 2),
        "test_score": round(100.0 * float(np.mean(test_scores)), 2),
        "delta": round(100.0 * float(np.mean(deltas)), 2),
        "changed_predictions": changed,
        "worst_ref_minus_test": worst,
        "best_test_minus_ref": best,
    }


def main() -> None:
    args = parse_args()
    if args.top_n <= 0:
        raise ValueError("--top_n must be positive.")

    tasks = _task_files(args.ref, args.test)
    task_reports = {
        task: compare_task(
            task,
            args.ref,
            args.test,
            top_n=args.top_n,
            allow_partial=bool(args.allow_partial),
        )
        for task in tasks
        if task in dataset2metric
    }
    ref_task_scores = {task: report["ref_score"] for task, report in task_reports.items()}
    test_task_scores = {task: report["test_score"] for task, report in task_reports.items()}
    ref_categories, ref_overall = aggregate_category_scores(ref_task_scores)
    test_categories, test_overall = aggregate_category_scores(test_task_scores)

    category_delta = {}
    for category, tasks_in_category in TASK_HIERARCHY.items():
        values = [task_reports[task]["delta"] for task in tasks_in_category if task in task_reports]
        if values:
            category_delta[category] = round(float(np.mean(values)), 2)

    payload = {
        "ref_dir": str(args.ref),
        "test_dir": str(args.test),
        "tasks": sorted(task_reports),
        "ref_category_scores": ref_categories,
        "test_category_scores": test_categories,
        "category_delta": category_delta,
        "ref_overall_category_avg": ref_overall,
        "test_overall_category_avg": test_overall,
        "overall_delta": None
        if ref_overall is None or test_overall is None
        else round(float(test_overall) - float(ref_overall), 2),
        "task_delta_sorted": sorted(
            (
                {
                    "task": task,
                    "ref_score": report["ref_score"],
                    "test_score": report["test_score"],
                    "delta": report["delta"],
                    "n": report["n"],
                    "changed_predictions": report["changed_predictions"],
                }
                for task, report in task_reports.items()
            ),
            key=lambda item: item["delta"],
        ),
        "task_reports": task_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(
        json.dumps(
            {
                "out": str(args.out),
                "ref_overall_category_avg": ref_overall,
                "test_overall_category_avg": test_overall,
                "overall_delta": payload["overall_delta"],
                "worst_tasks": payload["task_delta_sorted"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
