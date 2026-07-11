import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "profiling" / "bench_prefill_score.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_prefill_score", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(**overrides):
    values = {
        "variants": "three_pass_current,three_pass_host_bounds",
        "stages": "combined",
        "head_shapes": "qwen3_8b:32:8:128,qwen25_7b:28:4:128",
        "batch_sizes": "1",
        "context_lens": "4093",
        "windows": "32",
        "dtypes": "bf16",
        "score_dtypes": "fp32",
        "candidate_start": 64,
        "num_recent_tokens": 512,
        "slot_cases": "ordered",
        "layout_cases": "contiguous",
        "seed": 20260711,
        "warmup": 25,
        "rounds": 5,
        "iterations": 100,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_artifact_writer_creates_separate_required_outputs(tmp_path):
    module = _load_module()
    writer = module.ArtifactWriter(tmp_path / "run")
    assert {path.name for path in writer.run_dir.iterdir()} == set(module.ARTIFACT_NAMES)
    writer.write_json("run_info.json", {"seed": 20260711})
    writer.append_jsonl("raw_outputs.jsonl", {"case_id": "case", "rounds_ms": [1.0]})
    writer.append_jsonl("parsed_outputs.jsonl", {"case_id": "case", "latency_p50_ms": 1.0})
    writer.append_jsonl("per_sample_results.jsonl", {"case_id": "case", "status": "success"})
    assert json.loads((writer.run_dir / "run_info.json").read_text())["seed"] == 20260711
    assert json.loads((writer.run_dir / "per_sample_results.jsonl").read_text())["status"] == "success"


def test_manifest_is_deterministic_and_covers_both_model_geometries():
    module = _load_module()
    first = module.build_manifest(_args())
    second = module.build_manifest(_args())
    assert first == second
    assert len(first) == 4
    assert len({case.case_id for case in first}) == 4
    by_model = {case.model_shape: (case.Hq, case.Hkv, case.head_dim) for case in first}
    assert by_model == {"qwen3_8b": (32, 8, 128), "qwen25_7b": (28, 4, 128)}
    assert all(case.required and case.workspace_bytes > 0 for case in first)


def test_manifest_records_ragged_layout_and_launch_parameters():
    module = _load_module()
    manifest = module.build_manifest(
        _args(
            variants="three_pass_bn128",
            stages="partial,reduce,final,combined",
            head_shapes="qwen3_8b:32:8:128",
            batch_sizes="8",
            context_lens="32749",
            windows="128",
            slot_cases="shuffled",
            layout_cases="padded",
        )
    )
    assert len(manifest) == 4
    assert len(set(manifest[0].context_lens)) > 1
    assert len(set(manifest[0].score_windows)) > 1
    assert all(case.block_n == 128 and case.dot_stages == 3 for case in manifest)
    assert {case.kernel_launch_count for case in manifest if case.stage == "combined"} == {3}
    assert {case.kernel_launch_count for case in manifest if case.stage != "combined"} == {1}


def test_manifest_rejects_unknown_variant_and_too_short_context():
    module = _load_module()
    try:
        module.build_manifest(_args(variants="not-a-variant"))
    except ValueError as exc:
        assert "unknown variants" in str(exc)
    else:
        raise AssertionError("unknown variant must fail")
    try:
        module.build_manifest(_args(context_lens="512"))
    except ValueError as exc:
        assert "required minimum" in str(exc)
    else:
        raise AssertionError("short context must fail")


def test_aggregate_keeps_failures_and_separates_model_gates():
    module = _load_module()
    common = {
        "stage": "combined",
        "dtype": "bf16",
        "score_dtype": "fp32",
        "B": 1,
        "Hq": 32,
        "Hkv": 8,
        "gqa_ratio": 4,
        "head_dim": 128,
        "max_context_len": 4093,
        "context_lens": [4093],
        "score_windows": [32],
        "candidate_start": 64,
        "num_recent_tokens": 512,
        "slot_case": "ordered",
        "layout_case": "contiguous",
        "seed": 20260711,
        "kernel_launch_count": 3,
        "required": True,
    }
    rows = []
    for model_shape in ("qwen3_8b", "qwen25_7b"):
        rows.extend(
            [
                {
                    **common,
                    "case_id": f"{model_shape}-baseline",
                    "model_shape": model_shape,
                    "variant_id": "three_pass_current",
                    "status": "success",
                    "latency_p50_ms": 2.0,
                },
                {
                    **common,
                    "case_id": f"{model_shape}-candidate",
                    "model_shape": model_shape,
                    "variant_id": "three_pass_host_bounds",
                    "status": "success",
                    "latency_p50_ms": 1.0,
                },
            ]
        )
    rows.append(
        {
            **common,
            "case_id": "failed",
            "model_shape": "qwen3_8b",
            "variant_id": "three_pass_bn256",
            "status": "model_failed",
            "reason": "compile failed",
        }
    )
    aggregate = module.aggregate_results(rows)
    assert aggregate["status_counts"]["model_failed"] == 1
    assert not aggregate["all_required_success"]
    gates = [
        gate
        for gate in aggregate["performance_gates"]
        if gate["variant_id"] == "three_pass_host_bounds"
    ]
    assert {gate["model_shape"] for gate in gates} == {"qwen3_8b", "qwen25_7b"}
    assert all(gate["geomean_speedup"] == 2.0 and gate["passed"] for gate in gates)


def test_compile_metadata_uses_new_files_then_exact_signature_cache(tmp_path, monkeypatch):
    module = _load_module()
    cache_dir = tmp_path / "triton-cache"
    names = [
        "_prefill_score_partial_stats_kernel",
        "_prefill_score_reduce_stats_kernel",
        "_prefill_score_final_kernel",
    ]
    for name in names:
        stale = cache_dir / f"stale-{name}" / f"{name}.json"
        stale.parent.mkdir(parents=True)
        stale.write_text(json.dumps({"shared": 111}), encoding="utf-8")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(cache_dir))
    before = module._triton_metadata_snapshot()

    for name in names:
        exact = cache_dir / f"exact-{name}" / f"{name}.json"
        exact.parent.mkdir(parents=True)
        exact.write_text(
            json.dumps({"shared": 222, "num_warps": 4, "num_stages": 3}),
            encoding="utf-8",
        )
    case = module.build_manifest(
        _args(
            variants="three_pass_current",
            stages="combined",
            head_shapes="qwen3_8b:32:8:128",
        )
    )[0]

    resources, reason = module._compiled_resource_metadata(case, before)

    assert reason == ""
    assert [resource["shared_bytes"] for resource in resources] == [222, 222, 222]
    cached, reason = module._compiled_resource_metadata(
        case,
        module._triton_metadata_snapshot(),
    )
    assert reason == ""
    assert cached == resources
