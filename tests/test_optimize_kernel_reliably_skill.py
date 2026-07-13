import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).parents[1]
SKILL_ROOT = REPO_ROOT / ".agents" / "skills" / "optimize-kernel-reliably"
SCRIPTS = SKILL_ROOT / "scripts"


def _load_script(name):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(command):
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _init_run(tmp_path):
    run_dir = tmp_path / "run"
    result = _run(
        [
            sys.executable,
            str(SCRIPTS / "init_optimization_run.py"),
            "--run-dir",
            str(run_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--mode",
            "execute",
            "--target",
            "test-kernel",
            "--call-path",
            "wrapper -> test-kernel",
            "--source-path",
            str(SKILL_ROOT / "SKILL.md"),
            "--supported-input-domain",
            "BF16 contiguous rows with positive dimensions",
            "--hardware-scope",
            "test GPU scope",
            "--primary-metric",
            "latency_ms",
            "--direction",
            "minimize",
            "--workload-weight",
            "main=1.0",
            "--minimum-improvement",
            "0.03",
            "--maximum-case-regression",
            "0.05",
            "--max-candidates",
            "2",
            "--gpu-hours",
            "1",
            "--wall-hours",
            "2",
            "--allowed-path",
            ".agents/skills/optimize-kernel-reliably",
            "--commit-mode",
            "never",
        ]
    )
    assert result.returncode == 0, result.stderr
    return run_dir


def test_gpu_snapshot_parsers_preserve_device_and_process_evidence(monkeypatch):
    module = _load_script("check_gpu_idle.py")
    gpu_output = (
        "0, GPU-a, NVIDIA H100, 0, 10, 80000, 31, 70.5, 1200, 1500, P0, Disabled\n"
        "1, GPU-b, NVIDIA H100, 95, 40000, 80000, 70, 600, 1800, 1500, P0, Enabled\n"
    )
    process_output = "GPU-b, 123, python, 39000\n"
    gpus = module.parse_gpu_rows(gpu_output)
    processes = module.parse_process_rows(process_output)
    assert gpus[0]["uuid"] == "GPU-a"
    assert gpus[0]["utilization_gpu_pct"] == 0
    assert gpus[1]["memory_used_mib"] == 40000
    assert gpus[1]["mig_mode"] == "Enabled"
    assert processes == [
        {
            "gpu_uuid": "GPU-b",
            "pid": 123,
            "process_name": "python",
            "used_gpu_memory_mib": 39000,
        }
    ]
    outputs = iter((gpu_output, process_output))
    monkeypatch.setattr(module, "_run", lambda command: next(outputs))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    snapshot = module.take_snapshot("nvidia-smi", max_utilization=0, max_memory_used_mib=512)
    assert snapshot["selected"]["uuid"] == "GPU-a"
    assert snapshot["gpus"][1]["busy_reasons"] == [
        "utilization 95% > 0%",
        "memory 40000 MiB > 512 MiB",
        "1 compute process(es)",
    ]


def test_initialize_update_and_validate_control_run(tmp_path):
    run_dir = _init_run(tmp_path)
    validator = _load_script("validate_run_artifacts.py")
    assert validator.validate(run_dir) == []

    transition = _run(
        [
            sys.executable,
            str(SCRIPTS / "update_optimization_state.py"),
            "--run-dir",
            str(run_dir),
            "--to",
            "contract_frozen",
            "--evidence",
            "optimization_spec.json",
            "--pass-gate",
            "semantic_contract",
            "--next-action",
            "build the harness",
        ]
    )
    assert transition.returncode == 0, transition.stderr
    assert validator.validate(run_dir) == []
    state = json.loads((run_dir / "optimization_state.json").read_text())
    assert state["passed_gates"][0]["gate"] == "semantic_contract"

    result = {
        "schema_version": "1.0",
        "case_id": "negative-shape",
        "required": True,
        "variant_id": "baseline",
        "stage": "correctness",
        "status": "invalid_input",
        "failure_kind": "validation",
        "expected_status": "invalid_input",
        "gate_result": "pass",
        "attempt": 1,
        "first_attempt_ref": None,
        "reason": "shape rejected before launch",
    }
    (run_dir / "per_sample_results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
    assert validator.validate(run_dir) == []

    run_id = json.loads((run_dir / "optimization_spec.json").read_text())["run_id"]
    (run_dir / "decision.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "run_id": run_id,
                "decision": "blocked",
                "selected_variant": None,
                "reason": "test blocker",
                "gate_evidence": [],
                "qualification_manifest_hash": None,
                "source_hash": None,
                "rollback_point": None,
                "limitations": ["test only"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    conclude = _run(
        [
            sys.executable,
            str(SCRIPTS / "update_optimization_state.py"),
            "--run-dir",
            str(run_dir),
            "--to",
            "concluded",
            "--next-action",
            "none",
            "--stop-reason",
            "test blocker",
        ]
    )
    assert conclude.returncode == 0, conclude.stderr
    assert validator.validate(run_dir, require_concluded=True) == []


def test_state_machine_rejects_skipped_normal_state(tmp_path):
    run_dir = _init_run(tmp_path)
    result = _run(
        [
            sys.executable,
            str(SCRIPTS / "update_optimization_state.py"),
            "--run-dir",
            str(run_dir),
            "--to",
            "baseline_qualified",
            "--next-action",
            "invalid jump",
        ]
    )
    assert result.returncode != 0
    assert "invalid transition" in result.stderr


def test_validator_requires_registered_hash_for_changed_candidate_source(tmp_path):
    run_dir = _init_run(tmp_path)
    validator = _load_script("validate_run_artifacts.py")
    run_info_path = run_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text())
    source = tmp_path / "kernel.py"
    source.write_text("BASELINE = True\n", encoding="utf-8")
    source_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    run_info["source_hashes"] = source_hashes
    run_info["baseline_source_hash"] = validator._source_hash(source_hashes)
    run_info_path.write_text(json.dumps(run_info) + "\n", encoding="utf-8")
    assert validator.validate(run_dir) == []

    for state in ("contract_frozen", "harness_ready", "baseline_qualified", "profiled"):
        result = _run(
            [
                sys.executable,
                str(SCRIPTS / "update_optimization_state.py"),
                "--run-dir",
                str(run_dir),
                "--to",
                state,
                "--next-action",
                "continue test",
            ]
        )
        assert result.returncode == 0, result.stderr

    source.write_text("BASELINE = False\n", encoding="utf-8")
    errors = validator.validate(run_dir)
    assert "target source changed before candidate screening" in errors

    candidate_hashes = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    hypothesis = {
        "schema_version": "1.0",
        "variant_id": "candidate",
        "parent_variant_id": "baseline",
        "source_hash": validator._source_hash(candidate_hashes),
        "bottleneck_evidence": "test evidence",
        "primary_change": "test change",
        "predicted_win": "lower latency",
        "predicted_risk": "none",
        "correctness_impact": "none",
        "falsification_rule": "reject on regression",
        "status": "proposed",
        "evidence": [],
    }
    (run_dir / "hypotheses.jsonl").write_text(json.dumps(hypothesis) + "\n", encoding="utf-8")
    transition = _run(
        [
            sys.executable,
            str(SCRIPTS / "update_optimization_state.py"),
            "--run-dir",
            str(run_dir),
            "--to",
            "candidate_screening",
            "--candidate",
            "candidate",
            "--next-action",
            "screen candidate",
        ]
    )
    assert transition.returncode == 0, transition.stderr
    assert validator.validate(run_dir) == []


def test_paired_comparison_uses_all_pairs_and_fixed_bootstrap():
    module = _load_script("compare_paired_variants.py")
    rows = []
    for comparison_id, baseline_samples, candidate_samples in (
        ("short", [10.0, 10.2, 9.8], [8.0, 8.1, 7.9]),
        ("long", [20.0, 20.2, 19.8], [16.0, 16.1, 15.9]),
    ):
        for variant_id, samples, pair_order in (
            ("baseline", baseline_samples, ["baseline", "candidate"]),
            ("candidate", candidate_samples, ["baseline", "candidate"]),
        ):
            rows.append(
                {
                    "schema_version": "1.0",
                    "comparison_id": comparison_id,
                    "case_id": comparison_id,
                    "variant_id": variant_id,
                    "latency_samples_ms": samples,
                    "pair_order": pair_order,
                    "status": "success",
                    "weight": 1.0,
                }
            )
    result = module.compare(rows, "baseline", "candidate", 0.95, 1000, 7, 0.1, 0.0)
    assert result["passed"]
    assert result["case_count"] == 2
    assert result["weighted_geomean_latency_ratio"] == pytest.approx(0.8)
    assert result["worst_case_latency_ratio"] == pytest.approx(0.8)


def test_paired_comparison_rejects_missing_candidate_pair():
    module = _load_script("compare_paired_variants.py")
    row = {
        "schema_version": "1.0",
        "comparison_id": "only-baseline",
        "case_id": "case",
        "variant_id": "baseline",
        "latency_samples_ms": [1.0],
        "pair_order": ["baseline"],
        "status": "success",
    }
    with pytest.raises(ValueError, match="missing baseline/candidate pairs"):
        module.compare([row], "baseline", "candidate", 0.95, 100, 0, 0.01, 0.05)
