#!/usr/bin/env python3
"""Initialize an immutable kernel-optimization control run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


SCHEMA_VERSION = "1.0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(source_hashes: dict[str, str]) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def _atomic_json(path: Path, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def _parse_weights(values: list[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"workload weight must be BUCKET=WEIGHT, got {value!r}")
        bucket, raw_weight = value.split("=", 1)
        if not bucket or bucket in weights:
            raise ValueError(f"invalid or duplicate workload bucket {bucket!r}")
        weight = float(raw_weight)
        if weight <= 0:
            raise ValueError(f"workload weight must be positive, got {value!r}")
        weights[bucket] = weight
    return weights


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--framework",
        type=Path,
        default=Path(
            ".agents/skills/optimize-kernel-reliably/references/"
            "reliable-kernel-optimization-framework.md"
        ),
    )
    parser.add_argument("--mode", choices=("plan", "execute"), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--call-path", required=True)
    parser.add_argument("--source-path", action="append", required=True)
    parser.add_argument("--supported-input-domain", required=True)
    parser.add_argument("--hardware-scope", action="append", required=True)
    parser.add_argument("--primary-metric", required=True)
    parser.add_argument("--direction", choices=("minimize", "maximize"), required=True)
    parser.add_argument("--workload-weight", action="append", required=True)
    parser.add_argument("--minimum-improvement", type=float, required=True)
    parser.add_argument("--maximum-case-regression", type=float, required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--max-candidates", type=int, required=True)
    parser.add_argument("--gpu-hours", type=float, required=True)
    parser.add_argument("--wall-hours", type=float, required=True)
    parser.add_argument("--max-retries-per-case", type=int, default=1)
    parser.add_argument("--allowed-path", action="append", required=True)
    parser.add_argument("--commit-mode", choices=("never", "checkpoint", "final"), default="never")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    framework = args.framework if args.framework.is_absolute() else repo_root / args.framework
    if run_dir.exists():
        parser.error(f"run directory already exists: {run_dir}")
    if not framework.is_file():
        parser.error(f"framework file does not exist: {framework}")
    if not 0 < args.confidence_level < 1:
        parser.error("confidence level must be between 0 and 1")
    if min(args.minimum_improvement, args.maximum_case_regression) < 0:
        parser.error("performance thresholds must be non-negative")
    if args.max_candidates <= 0 or args.gpu_hours <= 0 or args.wall_hours <= 0:
        parser.error("candidate and time budgets must be positive")
    if args.max_retries_per_case < 0:
        parser.error("retry budget must be non-negative")
    try:
        weights = _parse_weights(args.workload_weight)
    except ValueError as exc:
        parser.error(str(exc))

    source_hashes: dict[str, str] = {}
    for raw_path in args.source_path:
        source = Path(raw_path)
        source = source if source.is_absolute() else repo_root / source
        if not source.is_file():
            parser.error(f"source path does not exist: {source}")
        try:
            relative = source.resolve().relative_to(repo_root)
        except ValueError:
            parser.error(f"source path must be inside the repository: {source}")
        source_hashes[str(relative)] = _sha256(source)

    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        framework_reference = str(framework.resolve().relative_to(repo_root))
    except ValueError:
        framework_reference = str(framework.resolve())
    run_identity = json.dumps(
        {"target": args.target, "timestamp": timestamp, "source_hashes": source_hashes},
        sort_keys=True,
    ).encode()
    run_id = f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{hashlib.sha256(run_identity).hexdigest()[:8]}"
    spec = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "frozen_at": timestamp,
        "mode": args.mode,
        "target": {
            "name": args.target,
            "call_path": args.call_path,
            "source_paths": list(source_hashes),
            "supported_input_domain": args.supported_input_domain,
        },
        "hardware_scope": args.hardware_scope,
        "objective": {
            "primary_metric": args.primary_metric,
            "direction": args.direction,
            "workload_weights": weights,
            "minimum_improvement": args.minimum_improvement,
            "maximum_case_regression": args.maximum_case_regression,
            "confidence_level": args.confidence_level,
        },
        "budget": {
            "max_candidates": args.max_candidates,
            "gpu_hours": args.gpu_hours,
            "wall_hours": args.wall_hours,
            "max_retries_per_case": args.max_retries_per_case,
        },
        "authorization": {
            "allowed_paths": args.allowed_path,
            "commit_mode": args.commit_mode,
        },
    }

    try:
        git_commit = _git(repo_root, "rev-parse", "HEAD")
        git_branch = _git(repo_root, "branch", "--show-current")
        git_status = _git(repo_root, "status", "--short")
    except RuntimeError as exc:
        parser.error(str(exc))

    run_dir.mkdir(parents=True, exist_ok=False)
    for directory in ("runs", "ncu", "nsys"):
        (run_dir / directory).mkdir()

    _atomic_json(run_dir / "optimization_spec.json", spec)
    spec_sha256 = _sha256(run_dir / "optimization_spec.json")
    state = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "state": "discovery",
        "previous_state": None,
        "updated_at": timestamp,
        "passed_gates": [],
        "failed_gates": [],
        "active_candidate": None,
        "candidate_status_counts": {"proposed": 0, "rejected": 0, "qualified": 0, "abandoned": 0},
        "budget_used": {"candidates": 0, "gpu_hours": 0.0, "wall_hours": 0.0},
        "next_action": "complete discovery and freeze the semantic contract",
        "stop_reason": None,
    }
    _atomic_json(run_dir / "optimization_state.json", state)
    _atomic_json(
        run_dir / "run_info.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "created_at": timestamp,
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "repo_root": str(repo_root),
            "git_commit": git_commit,
            "git_branch": git_branch,
            "git_status": git_status,
            "framework_path": framework_reference,
            "framework_sha256": _sha256(framework),
            "source_hashes": source_hashes,
            "baseline_source_hash": _source_hash(source_hashes),
            "optimization_spec_sha256": spec_sha256,
        },
    )
    _atomic_json(
        run_dir / "case_manifest.json",
        {"schema_version": SCHEMA_VERSION, "sets": {"development": None, "qualification": None, "integration": None}},
    )
    _atomic_json(run_dir / "aggregate_metrics.json", {"schema_version": SCHEMA_VERSION, "status": "pending"})
    (run_dir / "report.md").write_text(
        f"# Kernel optimization run {run_id}\n\nStatus: in progress.\n",
        encoding="utf-8",
    )
    for name in (
        "state_transitions.jsonl",
        "hypotheses.jsonl",
        "command_log.jsonl",
        "raw_outputs.jsonl",
        "parsed_outputs.jsonl",
        "per_sample_results.jsonl",
        "compile_metadata.jsonl",
    ):
        (run_dir / name).touch()
    _append_jsonl(
        run_dir / "state_transitions.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": timestamp,
            "kind": "initialize",
            "previous_state": None,
            "new_state": "discovery",
            "evidence": ["optimization_spec.json", "run_info.json"],
            "reason": "control run initialized",
        },
    )
    _append_jsonl(
        run_dir / "command_log.jsonl",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "started_at": timestamp,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "environment_delta": {},
            "return_code": 0,
        },
    )
    print(json.dumps({"run_id": run_id, "run_dir": str(run_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
