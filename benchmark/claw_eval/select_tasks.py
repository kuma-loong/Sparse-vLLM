from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml


VISUAL_FILE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".svg",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
}


class TaskSelectionError(RuntimeError):
    """Raised when a reproducible task selection cannot be created."""


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


def _task_files(task: dict[str, Any]) -> list[str]:
    prompt = task.get("prompt")
    attachments = prompt.get("attachments", []) if isinstance(prompt, dict) else []
    sandbox_files = task.get("sandbox_files", [])
    values = list(attachments or []) + list(sandbox_files or [])
    return sorted({str(value) for value in values})


def _skip_reasons(task: dict[str, Any]) -> list[str]:
    reasons = []
    if task.get("category") == "multimodal":
        reasons.append("category=multimodal")
    if "multimodal" in (task.get("tags") or []):
        reasons.append("tag=multimodal")
    visual_files = [
        value
        for value in _task_files(task)
        if Path(value).suffix.lower() in VISUAL_FILE_SUFFIXES
    ]
    if visual_files:
        reasons.append("visual_files=" + ",".join(visual_files))
    return reasons


def _ensure_selection_root(source_tasks_dir: Path, output_root: Path) -> Path:
    source_root = source_tasks_dir.parent.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    marker = output_root / ".claw_text_only_selection"
    existing = [path for path in output_root.iterdir() if path.name != marker.name]
    if existing and not marker.exists():
        raise TaskSelectionError(
            f"Refusing to reuse non-selection directory: {output_root}"
        )
    expected_marker = str(source_tasks_dir.resolve())
    if marker.exists() and marker.read_text(encoding="utf-8").strip() != expected_marker:
        raise TaskSelectionError(
            f"Selection directory belongs to another tasks directory: {output_root}"
        )
    marker.write_text(expected_marker + "\n", encoding="utf-8")

    for source in sorted(source_root.iterdir()):
        if source.name in {".git", source_tasks_dir.name}:
            continue
        if source.resolve() == output_root.resolve():
            continue
        target = output_root / source.name
        if target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve(), target_is_directory=source.is_dir())

    selected_tasks_dir = output_root / source_tasks_dir.name
    selected_tasks_dir.mkdir(exist_ok=True)
    return selected_tasks_dir


def select_text_only_tasks(
    *,
    source_tasks_dir: Path,
    output_root: Path,
    tag: str | None,
    summary_path: Path,
    skipped_results_path: Path,
) -> dict[str, Any]:
    if not source_tasks_dir.is_dir():
        raise TaskSelectionError(f"Tasks directory does not exist: {source_tasks_dir}")
    selected_tasks_dir = _ensure_selection_root(source_tasks_dir, output_root)
    selected = []
    skipped = []
    seen_ids = set()

    for task_yaml in sorted(source_tasks_dir.glob("*/task.yaml")):
        task = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(task, dict):
            raise TaskSelectionError(f"Task YAML must contain an object: {task_yaml}")
        task_id = task.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise TaskSelectionError(f"Task has no non-empty task_id: {task_yaml}")
        if task_id in seen_ids:
            raise TaskSelectionError(f"Duplicate task_id: {task_id}")
        seen_ids.add(task_id)
        tags = task.get("tags") or []
        if tag and tag not in tags:
            continue
        reasons = _skip_reasons(task)
        row = {
            "task_id": task_id,
            "task_name": task.get("task_name"),
            "category": task.get("category"),
            "task_yaml": str(task_yaml.resolve()),
        }
        if reasons:
            skipped.append({**row, "reasons": reasons})
            continue
        selected.append(row)
        target = selected_tasks_dir / task_yaml.parent.name
        source = task_yaml.parent.resolve()
        if target.is_symlink():
            if target.resolve() != source:
                raise TaskSelectionError(f"Selection symlink points to the wrong task: {target}")
        elif target.exists():
            raise TaskSelectionError(f"Selection target already exists and is not a symlink: {target}")
        else:
            target.symlink_to(source, target_is_directory=True)

    expected_names = {Path(row["task_yaml"]).parent.name for row in selected}
    stale = [path for path in selected_tasks_dir.iterdir() if path.name not in expected_names]
    if stale:
        raise TaskSelectionError(
            "Selection directory contains stale task links: "
            + ", ".join(str(path) for path in stale[:10])
        )
    if not selected:
        raise TaskSelectionError("Text-only task selection is empty")

    skipped_rows = [
        {
            "task_id": row["task_id"],
            "status": "skipped_by_policy",
            "resolved": None,
            "score": None,
            "trials": 0,
            "error": None,
            "skip_reason": "; ".join(row["reasons"]),
        }
        for row in skipped
    ]
    summary = {
        "schema_version": 1,
        "policy": "text_only_no_visual_files",
        "excluded_categories": ["multimodal"],
        "excluded_file_suffixes": sorted(VISUAL_FILE_SUFFIXES),
        "tag": tag,
        "source_tasks_dir": str(source_tasks_dir.resolve()),
        "selected_tasks_dir": str(selected_tasks_dir.resolve()),
        "selected_count": len(selected),
        "skipped_count": len(skipped),
        "selected": selected,
        "skipped": skipped,
    }
    _write_json(summary_path, summary)
    _write_jsonl(skipped_results_path, skipped_rows)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Claw-Eval task view safe for a text-only model"
    )
    parser.add_argument("--source-tasks-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--skipped-results", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = select_text_only_tasks(
            source_tasks_dir=args.source_tasks_dir,
            output_root=args.output_root,
            tag=args.tag,
            summary_path=args.summary,
            skipped_results_path=args.skipped_results,
        )
    except (OSError, yaml.YAMLError, TaskSelectionError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    print(
        f"Selected {summary['selected_count']} text-only task(s); "
        f"skipped {summary['skipped_count']} task(s) by policy"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
