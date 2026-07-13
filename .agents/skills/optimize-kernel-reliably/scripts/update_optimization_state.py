#!/usr/bin/env python3
"""Atomically update a kernel-optimization state machine."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


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
DECISIONS = {"selected", "no_improvement", "inconclusive", "blocked"}


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--to", choices=STATES, required=True)
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--candidate")
    parser.add_argument("--pass-gate", action="append", default=[])
    parser.add_argument("--fail-gate", action="append", default=[])
    parser.add_argument("--next-action", required=True)
    parser.add_argument("--stop-reason")
    parser.add_argument("--candidates-used", type=int)
    parser.add_argument("--gpu-hours-used", type=float)
    parser.add_argument("--wall-hours-used", type=float)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    state_path = run_dir / "optimization_state.json"
    spec = _load_json(run_dir / "optimization_spec.json")
    state = _load_json(state_path)
    current = str(state["state"])
    if current == "concluded":
        parser.error("a concluded run is immutable")
    current_index = STATES.index(current)
    target_index = STATES.index(args.to)
    if args.to != current and args.to != "concluded" and target_index != current_index + 1:
        parser.error(f"invalid transition {current} -> {args.to}")
    duplicate_gates = set(args.pass_gate) & set(args.fail_gate)
    if duplicate_gates:
        parser.error(f"gates cannot pass and fail in the same update: {sorted(duplicate_gates)}")
    if args.fail_gate and args.to not in {current, "concluded"}:
        parser.error("a failed gate cannot advance the state machine")

    evidence = []
    for raw_path in args.evidence:
        path = Path(raw_path)
        path = path if path.is_absolute() else run_dir / path
        if not path.exists():
            parser.error(f"evidence does not exist: {path}")
        try:
            evidence.append(str(path.resolve().relative_to(run_dir)))
        except ValueError:
            evidence.append(str(path.resolve()))
    if (args.pass_gate or args.fail_gate) and not evidence:
        parser.error("gate updates require at least one --evidence path")

    if args.to == "qualification_locked":
        manifest = _load_json(run_dir / "case_manifest.json")
        qualification = manifest.get("sets", {}).get("qualification")
        if not isinstance(qualification, dict) or qualification.get("exposed") is not False:
            parser.error("qualification_locked requires an indexed, unexposed qualification manifest")
        qualification_path = Path(str(qualification.get("path", "")))
        if not qualification_path.is_absolute():
            qualification_path = run_dir / qualification_path
        if not qualification_path.is_file():
            parser.error(f"qualification manifest does not exist: {qualification_path}")
        if _sha256(qualification_path) != qualification.get("sha256"):
            parser.error("qualification manifest hash does not match case_manifest.json")

    decision = None
    if args.to == "concluded":
        decision_path = run_dir / "decision.json"
        if not decision_path.is_file():
            parser.error("concluded requires decision.json")
        decision = _load_json(decision_path).get("decision")
        if decision not in DECISIONS:
            parser.error(f"invalid decision: {decision!r}")
        if decision == "selected" and current != "integration":
            parser.error("selected may conclude only from integration")
        if not args.stop_reason:
            parser.error("concluded requires --stop-reason")

    budget = spec["budget"]
    used = dict(state["budget_used"])
    updates = {
        "candidates": args.candidates_used,
        "gpu_hours": args.gpu_hours_used,
        "wall_hours": args.wall_hours_used,
    }
    limits = {
        "candidates": budget["max_candidates"],
        "gpu_hours": budget["gpu_hours"],
        "wall_hours": budget["wall_hours"],
    }
    for key, value in updates.items():
        if value is None:
            continue
        if value < used[key]:
            parser.error(f"{key} usage cannot decrease")
        if value > limits[key] and args.to != "concluded":
            parser.error(f"{key} usage {value} exceeds budget {limits[key]}")
        if value > limits[key] and decision not in {"inconclusive", "blocked"}:
            parser.error(f"{key} budget may be exceeded only by an inconclusive or blocked decision")
        used[key] = value

    timestamp = datetime.now(timezone.utc).isoformat()
    active_candidate = args.candidate if args.candidate is not None else state["active_candidate"]
    passed_gates = list(state["passed_gates"])
    failed_gates = list(state["failed_gates"])
    gate_updates = [(gate, passed_gates) for gate in args.pass_gate]
    gate_updates.extend((gate, failed_gates) for gate in args.fail_gate)
    for gate, records in gate_updates:
        records.append(
            {
                "gate": gate,
                "timestamp": timestamp,
                "candidate": active_candidate,
                "evidence": evidence,
            }
        )
    event = {
        "schema_version": "1.0",
        "run_id": state["run_id"],
        "timestamp": timestamp,
        "kind": "update" if args.to == current else "transition",
        "previous_state": current,
        "new_state": args.to,
        "evidence": evidence,
        "candidate": active_candidate,
        "passed_gates": args.pass_gate,
        "failed_gates": args.fail_gate,
        "budget_used": used,
        "next_action": args.next_action,
        "reason": args.stop_reason,
    }
    with (run_dir / "state_transitions.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")

    state.update(
        {
            "previous_state": current,
            "state": args.to,
            "updated_at": timestamp,
            "active_candidate": active_candidate,
            "passed_gates": passed_gates,
            "failed_gates": failed_gates,
            "budget_used": used,
            "next_action": args.next_action,
            "stop_reason": args.stop_reason,
        }
    )
    _atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
