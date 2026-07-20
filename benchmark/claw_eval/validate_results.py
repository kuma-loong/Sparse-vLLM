from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


class ClawResultError(RuntimeError):
    """Raised when Claw-Eval artifacts are incomplete or contain task errors."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ClawResultError(f"Required Claw-Eval artifact is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ClawResultError(f"Invalid JSON in Claw-Eval artifact {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _read_skipped_results(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ClawResultError(f"Skipped-results artifact is missing: {path}") from exc
    rows = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ClawResultError(
                f"Invalid JSON on line {line_number} of skipped-results artifact {path}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ClawResultError(
                f"Skipped-results line {line_number} must be a JSON object: {path}"
            )
        if row.get("status") != "skipped_by_policy":
            raise ClawResultError(
                f"Skipped-results line {line_number} has invalid status: {row.get('status')!r}"
            )
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ClawResultError(
                f"Skipped-results line {line_number} has no non-empty task_id: {path}"
            )
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_state(trace_dir: Path) -> dict[str, dict[str, Any]]:
    if not trace_dir.exists():
        return {}
    state = {}
    for path in sorted(trace_dir.rglob("batch_summary.json")):
        stat = path.stat()
        state[str(path.relative_to(trace_dir))] = {
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256(path),
            "size": stat.st_size,
        }
    return state


def write_snapshot(trace_dir: Path, snapshot_path: Path) -> None:
    _write_json(
        snapshot_path,
        {
            "trace_dir": str(trace_dir.resolve()),
            "batch_summaries": _artifact_state(trace_dir),
        },
    )


def _changed_summary(trace_dir: Path, snapshot_path: Path) -> Path:
    snapshot = _read_json(snapshot_path)
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("batch_summaries"), dict):
        raise ClawResultError(f"Invalid Claw-Eval pre-run snapshot: {snapshot_path}")
    if snapshot.get("trace_dir") != str(trace_dir.resolve()):
        raise ClawResultError(
            f"Claw-Eval snapshot trace directory does not match {trace_dir}: {snapshot_path}"
        )

    before = snapshot["batch_summaries"]
    after = _artifact_state(trace_dir)
    changed = [relative for relative, state in after.items() if before.get(relative) != state]
    if len(changed) != 1:
        raise ClawResultError(
            "Expected exactly one changed batch result after Claw-Eval, "
            f"found {len(changed)}: {changed}"
        )
    return trace_dir / changed[0]


def _normalize_task(result: Any, expected_trials: int) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ClawResultError("Claw-Eval batch_results entries must be JSON objects")
    task_id = result.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ClawResultError("Claw-Eval result has no non-empty task_id")

    top_error = result.get("error")
    trials = result.get("trials")
    errors = []
    if top_error:
        errors.append(str(top_error))
    if not isinstance(trials, list):
        errors.append("trials is not a list")
        trials = []
    if len(trials) != expected_trials:
        errors.append(f"expected {expected_trials} trials, found {len(trials)}")

    valid_scores = []
    passed_values = []
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            errors.append(f"trial {index + 1} is not an object")
            continue
        if trial.get("error"):
            errors.append(f"trial {index + 1}: {trial['error']}")
        passed = trial.get("passed")
        score = trial.get("task_score")
        if not isinstance(passed, bool):
            errors.append(f"trial {index + 1} has invalid passed value")
        else:
            passed_values.append(passed)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            errors.append(f"trial {index + 1} has invalid task_score")
        else:
            valid_scores.append(float(score))

    status = "metric_failed" if errors else "success"
    return {
        "task_id": task_id,
        "status": status,
        "resolved": all(passed_values) if status == "success" else None,
        "score": (
            sum(valid_scores) / len(valid_scores)
            if status == "success" and valid_scores
            else None
        ),
        "trials": len(trials),
        "error": "; ".join(errors) if errors else None,
    }


def validate_changed_results(
    *,
    trace_dir: Path,
    snapshot_path: Path,
    per_sample_path: Path,
    final_summary_path: Path,
    skipped_results_path: Path | None = None,
) -> dict[str, Any]:
    summary_path = _changed_summary(trace_dir, snapshot_path)
    results_path = summary_path.with_name("batch_results.json")
    upstream_summary = _read_json(summary_path)
    results = _read_json(results_path)
    if not isinstance(upstream_summary, dict):
        raise ClawResultError(f"Claw-Eval summary must be a JSON object: {summary_path}")
    if not isinstance(results, list):
        raise ClawResultError(f"Claw-Eval results must be a JSON list: {results_path}")

    tasks = upstream_summary.get("tasks")
    trials_per_task = upstream_summary.get("trials_per_task")
    if not isinstance(tasks, int) or isinstance(tasks, bool) or tasks <= 0:
        raise ClawResultError(f"Claw-Eval summary has invalid task count: {tasks!r}")
    if (
        not isinstance(trials_per_task, int)
        or isinstance(trials_per_task, bool)
        or trials_per_task <= 0
    ):
        raise ClawResultError(
            f"Claw-Eval summary has invalid trials_per_task: {trials_per_task!r}"
        )
    if len(results) != tasks:
        raise ClawResultError(
            f"Claw-Eval result count {len(results)} does not match summary tasks {tasks}"
        )

    rows = [_normalize_task(result, trials_per_task) for result in results]
    skipped_rows = _read_skipped_results(skipped_results_path)
    task_ids = [row["task_id"] for row in rows]
    if len(set(task_ids)) != len(task_ids):
        raise ClawResultError("Claw-Eval results contain duplicate task_id values")
    skipped_task_ids = [row["task_id"] for row in skipped_rows]
    if len(set(skipped_task_ids)) != len(skipped_task_ids):
        raise ClawResultError("Skipped-results artifact contains duplicate task_id values")
    overlap = sorted(set(task_ids) & set(skipped_task_ids))
    if overlap:
        raise ClawResultError(
            f"Tasks cannot be both evaluated and skipped_by_policy: {overlap}"
        )
    all_rows = rows + skipped_rows

    top_level_errors = sum(
        1 for result in results if isinstance(result, dict) and result.get("error")
    )
    if upstream_summary.get("errored") != top_level_errors:
        raise ClawResultError(
            "Claw-Eval summary errored count does not match batch_results: "
            f"{upstream_summary.get('errored')!r} != {top_level_errors}"
        )

    status_counts = dict(sorted(Counter(row["status"] for row in all_rows).items()))
    normalized_summary = {
        "schema_version": 1,
        "tasks": tasks,
        "skipped_tasks": len(skipped_rows),
        "total_scope_tasks": tasks + len(skipped_rows),
        "trials_per_task": trials_per_task,
        "resolved_tasks": sum(row["resolved"] is True for row in rows),
        "status_counts": status_counts,
        "failed_task_ids": [row["task_id"] for row in rows if row["status"] != "success"],
        "skipped_task_ids": skipped_task_ids,
        "batch_results_path": str(results_path.resolve()),
        "batch_summary_path": str(summary_path.resolve()),
        "skipped_results_path": (
            str(skipped_results_path.resolve()) if skipped_results_path is not None else None
        ),
        "upstream_summary": upstream_summary,
    }
    _write_jsonl(per_sample_path, all_rows)
    _write_json(final_summary_path, normalized_summary)

    failed = sum(row["status"] != "success" for row in rows)
    if failed:
        raise ClawResultError(
            f"Claw-Eval completed with {failed} task evaluation failure(s); "
            f"see {per_sample_path}"
        )
    return normalized_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Claw-Eval batch artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--trace-dir", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--trace-dir", type=Path, required=True)
    validate.add_argument("--snapshot", type=Path, required=True)
    validate.add_argument("--per-sample", type=Path, required=True)
    validate.add_argument("--final-summary", type=Path, required=True)
    validate.add_argument("--skipped-results", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            write_snapshot(args.trace_dir, args.output)
        else:
            summary = validate_changed_results(
                trace_dir=args.trace_dir,
                snapshot_path=args.snapshot,
                per_sample_path=args.per_sample,
                final_summary_path=args.final_summary,
                skipped_results_path=args.skipped_results,
            )
            print(
                "Claw-Eval artifacts validated: "
                f"tasks={summary['tasks']} resolved={summary['resolved_tasks']}"
            )
    except ClawResultError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 7
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
