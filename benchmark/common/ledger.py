from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from benchmark.common.paths import benchmark_output_root


STATUS_VALUES = {
    "success",
    "invalid_run",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
    "oom",
    "timeout",
}


LEDGER_COLUMNS = [
    "run_id",
    "timestamp",
    "feature",
    "objective",
    "git_commit",
    "dirty",
    "branch",
    "benchmark",
    "benchmark_tier",
    "benchmark_source",
    "script",
    "command",
    "model_path",
    "tokenizer_path",
    "method",
    "method_config",
    "baseline_run_id",
    "previous_run_id",
    "dataset",
    "split",
    "sample_policy",
    "sample_ids",
    "lengths",
    "max_new_tokens",
    "decode_config",
    "gpu",
    "env",
    "output_dir",
    "status",
    "primary_metrics",
    "quality_delta",
    "speedup",
    "memory_delta",
    "failure_summary",
    "decision",
    "notes",
]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_metadata(cwd: str | Path) -> dict[str, Any]:
    cwd = str(cwd)

    def run_git(args: list[str]) -> str:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()

    try:
        commit = run_git(["rev-parse", "--short=12", "HEAD"])
    except Exception:
        commit = "unknown"
    try:
        branch = run_git(["branch", "--show-current"]) or "detached"
    except Exception:
        branch = "unknown"
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True).strip())
    except Exception:
        dirty = True
    return {"git_commit": commit, "branch": branch, "dirty": dirty}


def default_ledger_paths(feature: str, output_root: str | Path | None = None) -> tuple[Path, Path]:
    root = Path(output_root).expanduser() if output_root else benchmark_output_root()
    ledger_root = root / "_ledgers"
    return ledger_root / f"{feature}.jsonl", ledger_root / f"{feature}.csv"


def _encode_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, bool)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def append_ledger_record(
    record: Mapping[str, Any],
    *,
    jsonl_path: str | Path,
    csv_path: str | Path | None = None,
) -> None:
    status = record.get("status")
    if status not in STATUS_VALUES:
        raise ValueError(f"Invalid benchmark ledger status {status!r}; expected one of {sorted(STATUS_VALUES)}")

    jsonl_path = Path(jsonl_path).expanduser()
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True) + "\n")

    if csv_path is None:
        return
    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({column: _encode_cell(record.get(column)) for column in LEDGER_COLUMNS})


def selected_env_snapshot() -> dict[str, str]:
    names = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "SVLLM_BENCHMARK_OUTPUT_DIR",
        "SVLLM_BENCHMARK_DATA_DIR",
        "SVLLM_LONGBENCH_DATA_DIR",
        "SVLLM_SCBENCH_PREPROCESSED_ROOT",
        "DELTAKV_OUTPUT_DIR",
        "DELTAKV_OUTPUT_BASE",
        "DELTAKV_DATA_DIR",
        "DELTAKV_LONGBENCH_DATA_DIR",
        "SCBENCH_PREPROCESSED_ROOT",
        "http_proxy",
        "https_proxy",
    ]
    return {name: value for name in names if (value := os.getenv(name))}
