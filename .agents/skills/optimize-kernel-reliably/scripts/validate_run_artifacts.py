#!/usr/bin/env python3
"""Validate a reliable kernel-optimization control run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCHEMA_VERSION = "1.0"
VALID_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
VALID_DECISIONS = {"selected", "no_improvement", "inconclusive", "blocked"}
STATES = (
    "discovery",
    "contract_frozen",
    "harness_ready",
    "baseline_qualified",
    "profiled",
    "candidate_screening",
    "qualification_locked",
    "integration",
    "concluded",
)
REQUIRED_FILES = {
    "optimization_spec.json",
    "optimization_state.json",
    "state_transitions.jsonl",
    "run_info.json",
    "case_manifest.json",
    "hypotheses.jsonl",
    "command_log.jsonl",
    "raw_outputs.jsonl",
    "parsed_outputs.jsonl",
    "per_sample_results.jsonl",
    "aggregate_metrics.json",
    "compile_metadata.jsonl",
    "report.md",
}
JSONL_FILES = {
    "state_transitions.jsonl",
    "hypotheses.jsonl",
    "command_log.jsonl",
    "raw_outputs.jsonl",
    "parsed_outputs.jsonl",
    "per_sample_results.jsonl",
    "compile_metadata.jsonl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hash(source_hashes: dict[str, str]) -> str:
    payload = json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path, errors: list[str]) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{path.name}: invalid JSON: {exc}")
        return None
    if not isinstance(value, dict):
        errors.append(f"{path.name}: expected a JSON object")
        return None
    if value.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"{path.name}: schema_version must be {SCHEMA_VERSION!r}")
    return value


def _load_jsonl(path: Path, errors: list[str]) -> list[dict[str, object]]:
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(f"{path.name}: cannot read: {exc}")
        return rows
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}:{line_number}: invalid JSON: {exc}")
            continue
        if not isinstance(value, dict):
            errors.append(f"{path.name}:{line_number}: expected a JSON object")
            continue
        if value.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"{path.name}:{line_number}: schema_version must be {SCHEMA_VERSION!r}")
        rows.append(value)
    return rows


def _resolve_recorded_path(path_value: object, run_dir: Path, repo_root: Path) -> Path:
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    run_relative = run_dir / path
    return run_relative if run_relative.exists() else repo_root / path


def validate(run_dir: Path, require_concluded: bool = False) -> list[str]:
    errors: list[str] = []
    run_dir = run_dir.resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (run_dir / name).is_file())
    errors.extend(f"missing required artifact: {name}" for name in missing)
    if missing:
        return errors

    spec = _load_json(run_dir / "optimization_spec.json", errors)
    state = _load_json(run_dir / "optimization_state.json", errors)
    run_info = _load_json(run_dir / "run_info.json", errors)
    manifest = _load_json(run_dir / "case_manifest.json", errors)
    _load_json(run_dir / "aggregate_metrics.json", errors)
    jsonl = {name: _load_jsonl(run_dir / name, errors) for name in JSONL_FILES}
    if spec is None or state is None or run_info is None or manifest is None:
        return errors

    run_id = spec.get("run_id")
    for name, value in (("optimization_state.json", state), ("run_info.json", run_info)):
        if value.get("run_id") != run_id:
            errors.append(f"{name}: run_id does not match optimization_spec.json")

    if run_info.get("optimization_spec_sha256") != _sha256(run_dir / "optimization_spec.json"):
        errors.append("optimization_spec.json hash does not match run_info.json")

    repo_root = Path(str(run_info.get("repo_root", run_dir))).resolve()
    framework_path = _resolve_recorded_path(run_info.get("framework_path"), run_dir, repo_root)
    if not framework_path.is_file():
        errors.append(f"recorded framework does not exist: {framework_path}")
    elif run_info.get("framework_sha256") != _sha256(framework_path):
        errors.append("framework hash changed after run initialization")

    source_hashes = run_info.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        errors.append("run_info.json: source_hashes must be a non-empty object")
    else:
        current_source_hashes = {}
        for recorded_path, expected_hash in source_hashes.items():
            source_path = _resolve_recorded_path(recorded_path, run_dir, repo_root)
            if not source_path.is_file():
                errors.append(f"recorded source does not exist: {source_path}")
                continue
            current_source_hashes[recorded_path] = _sha256(source_path)
        baseline_source_hash = _source_hash(source_hashes)
        if run_info.get("baseline_source_hash") != baseline_source_hash:
            errors.append("run_info.json: baseline_source_hash does not match source_hashes")
        if len(current_source_hashes) == len(source_hashes):
            current_source_hash = _source_hash(current_source_hashes)
            if current_source_hash != baseline_source_hash:
                active_candidate = state.get("active_candidate")
                candidate_rows = [
                    row
                    for row in jsonl["hypotheses.jsonl"]
                    if row.get("variant_id") == active_candidate
                ]
                if state.get("state") not in {
                    "candidate_screening",
                    "qualification_locked",
                    "integration",
                    "concluded",
                }:
                    errors.append("target source changed before candidate screening")
                elif not active_candidate or not candidate_rows:
                    errors.append("target source changed without an active registered candidate")
                elif candidate_rows[-1].get("source_hash") != current_source_hash:
                    errors.append("active candidate source hash does not match the working tree")

    transitions = jsonl["state_transitions.jsonl"]
    if not transitions:
        errors.append("state_transitions.jsonl: at least one transition is required")
    else:
        transition_state = None
        for index, transition in enumerate(transitions, 1):
            previous_state = transition.get("previous_state")
            new_state = transition.get("new_state")
            if previous_state != transition_state:
                errors.append(
                    f"state_transitions.jsonl:{index}: previous_state does not match prior event"
                )
            if new_state not in STATES:
                errors.append(f"state_transitions.jsonl:{index}: invalid state {new_state!r}")
                continue
            if transition_state is not None and new_state not in {transition_state, "concluded"}:
                if STATES.index(new_state) != STATES.index(transition_state) + 1:
                    errors.append(f"state_transitions.jsonl:{index}: skipped a normal state")
            transition_state = new_state
        if transition_state != state.get("state"):
            errors.append("optimization_state.json does not match the last state transition")

    for index, row in enumerate(jsonl["hypotheses.jsonl"], 1):
        required_fields = (
            "variant_id",
            "parent_variant_id",
            "source_hash",
            "bottleneck_evidence",
            "primary_change",
            "predicted_win",
            "predicted_risk",
            "correctness_impact",
            "falsification_rule",
            "status",
            "evidence",
        )
        absent = [field for field in required_fields if field not in row]
        if absent:
            errors.append(f"hypotheses.jsonl:{index}: missing fields {absent}")
        if row.get("status") not in {"proposed", "rejected", "qualified", "abandoned"}:
            errors.append(f"hypotheses.jsonl:{index}: invalid status {row.get('status')!r}")

    sets = manifest.get("sets")
    if not isinstance(sets, dict) or set(sets) != {"development", "qualification", "integration"}:
        errors.append("case_manifest.json: sets must contain development, qualification, and integration")
        sets = {}
    for set_name, entry in sets.items():
        if entry is None:
            continue
        if not isinstance(entry, dict):
            errors.append(f"case_manifest.json: {set_name} must be null or an object")
            continue
        for key in ("path", "sha256", "exposed"):
            if key not in entry:
                errors.append(f"case_manifest.json: {set_name} is missing {key}")
        if "path" not in entry or "sha256" not in entry:
            continue
        set_path = _resolve_recorded_path(entry["path"], run_dir, repo_root)
        if not set_path.is_file():
            errors.append(f"manifest set does not exist: {set_path}")
        elif _sha256(set_path) != entry["sha256"]:
            errors.append(f"manifest hash changed: {set_name}")

    seen_results: set[tuple[object, ...]] = set()
    for index, row in enumerate(jsonl["per_sample_results.jsonl"], 1):
        required_fields = (
            "case_id",
            "required",
            "variant_id",
            "stage",
            "status",
            "failure_kind",
            "expected_status",
            "gate_result",
            "attempt",
            "first_attempt_ref",
            "reason",
        )
        absent = [field for field in required_fields if field not in row]
        if absent:
            errors.append(f"per_sample_results.jsonl:{index}: missing fields {absent}")
            continue
        if row["status"] not in VALID_STATUSES:
            errors.append(f"per_sample_results.jsonl:{index}: invalid status {row['status']!r}")
        if row["expected_status"] not in VALID_STATUSES:
            errors.append(f"per_sample_results.jsonl:{index}: invalid expected_status {row['expected_status']!r}")
        if row["gate_result"] not in {"pass", "fail"}:
            errors.append(f"per_sample_results.jsonl:{index}: gate_result must be pass or fail")
        if not isinstance(row["attempt"], int) or row["attempt"] < 1:
            errors.append(f"per_sample_results.jsonl:{index}: attempt must be a positive integer")
        identity = (row["case_id"], row["variant_id"], row["stage"], row["attempt"])
        if identity in seen_results:
            errors.append(f"per_sample_results.jsonl:{index}: duplicate result identity {identity}")
        seen_results.add(identity)

    concluded = state.get("state") == "concluded"
    if require_concluded and not concluded:
        errors.append("run is not concluded")
    if concluded:
        decision_path = run_dir / "decision.json"
        if not decision_path.is_file():
            errors.append("concluded run is missing decision.json")
        else:
            decision = _load_json(decision_path, errors)
            if decision is not None:
                required_decision_fields = (
                    "run_id",
                    "decision",
                    "selected_variant",
                    "reason",
                    "gate_evidence",
                    "qualification_manifest_hash",
                    "source_hash",
                    "rollback_point",
                    "limitations",
                )
                missing_decision_fields = [
                    field for field in required_decision_fields if field not in decision
                ]
                if missing_decision_fields:
                    errors.append(f"decision.json: missing fields {missing_decision_fields}")
                if decision.get("run_id") != run_id:
                    errors.append("decision.json: run_id does not match")
                if decision.get("decision") not in VALID_DECISIONS:
                    errors.append(f"decision.json: invalid decision {decision.get('decision')!r}")
                if decision.get("decision") == "selected":
                    qualification = sets.get("qualification")
                    if not isinstance(qualification, dict) or qualification.get("exposed") is not False:
                        errors.append("selected decision requires an unexposed qualification manifest")
                    integration = sets.get("integration")
                    if not isinstance(integration, dict):
                        errors.append("selected decision requires an integration manifest")
                    if not decision.get("selected_variant") or not decision.get("source_hash"):
                        errors.append("selected decision requires selected_variant and source_hash")
                    if not decision.get("gate_evidence"):
                        errors.append("selected decision requires gate_evidence")
        failed_required = [
            row
            for row in jsonl["per_sample_results.jsonl"]
            if row.get("required") is True and row.get("gate_result") != "pass"
        ]
        if failed_required and (run_dir / "decision.json").is_file():
            decision = _load_json(run_dir / "decision.json", [])
            if decision and decision.get("decision") in {"selected", "no_improvement"}:
                errors.append("selected or no_improvement decision contains failed required samples")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--require-concluded", action="store_true")
    args = parser.parse_args()
    errors = validate(args.run_dir, require_concluded=args.require_concluded)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"validated kernel optimization run: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
