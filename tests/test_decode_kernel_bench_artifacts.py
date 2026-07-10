import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).parents[1] / "scripts" / "profiling" / "kernel_bench" / "bench_gqa_decode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bench_gqa_decode", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_writer_creates_separate_required_outputs(tmp_path):
    module = _load_module()
    writer = module.ArtifactWriter(tmp_path / "run")
    assert {path.name for path in writer.run_dir.iterdir()} >= set(module.ARTIFACT_NAMES) | {"ncu", "nsys"}
    writer.write_json("run_info.json", {"seed": 20260710})
    writer.append_jsonl("raw_outputs.jsonl", {"case_id": "case", "rounds_ms": [1.0]})
    writer.append_jsonl("parsed_outputs.jsonl", {"case_id": "case", "latency_p50_ms": 1.0})
    writer.append_jsonl("per_sample_results.jsonl", {"case_id": "case", "status": "success"})
    assert json.loads((writer.run_dir / "run_info.json").read_text())["seed"] == 20260710
    assert json.loads((writer.run_dir / "per_sample_results.jsonl").read_text())["status"] == "success"


def test_manifest_uses_stable_ids_and_complete_case_fields():
    module = _load_module()
    args = SimpleNamespace(
        variants="grouped_s1_allow256_w4,per_q_s1_w8,unified_s2_w8",
        stages="stage1,stage2,combined",
        dtypes="bf16",
        score_modes="none",
        batch_sizes="1",
        seq_lens="256",
        head_dims="256",
        block_seqs="256",
        slot_orders="ordered",
        num_heads=16,
        num_kv_heads=4,
        seed=20260710,
        warmup=25,
        rounds=5,
        iterations=100,
    )
    first = module.build_manifest(args)
    second = module.build_manifest(args)
    assert first == second
    assert len(first) == 5
    assert len({case.case_id for case in first}) == 5
    assert all(case.required and case.gqa_ratio == 4 for case in first)


def test_long_context_manifest_contains_baseline_and_optimized_data():
    module = _load_module()
    args = SimpleNamespace(
        variants="per_q_s1_w8,grouped_s1_bn16_w2_s2",
        stages="stage1",
        dtypes="bf16",
        score_modes="none",
        batch_sizes="1",
        seq_lens="131072,262144",
        head_dims="256",
        block_seqs="256",
        slot_orders="ordered",
        num_heads=16,
        num_kv_heads=4,
        seed=20260710,
        warmup=25,
        rounds=5,
        iterations=100,
    )
    manifest = module.build_manifest(args)
    assert len(manifest) == 4
    assert {case.context_len for case in manifest} == {131072, 262144}
    assert {case.variant_id for case in manifest} == {"per_q_s1_w8", "grouped_s1_bn16_w2_s2"}
    assert all(case.block_n == 16 and case.num_stages == 2 for case in manifest)


def test_broad_profile_covers_boundaries_and_realistic_irregular_lengths():
    module = _load_module()
    args = SimpleNamespace(
        variants="per_q_s1_w8,grouped_s1_bn16_w2_s2",
        stages="combined",
        dtypes="bf16",
        score_modes="none",
        batch_sizes="1",
        seq_lens=None,
        seq_profile="broad",
        head_dims="256",
        block_seqs="512",
        slot_orders="ordered",
        num_heads=16,
        num_kv_heads=4,
        seed=20260710,
        warmup=25,
        rounds=5,
        iterations=100,
    )
    manifest = module.build_manifest(args)
    lengths = {case.context_len for case in manifest}
    assert {2535, 6872, 79439} <= lengths
    assert {15, 16, 17, 127, 128, 129, 511, 512, 513} <= lengths
    assert {4093, 4096, 32749, 65521, 65536, 131071, 262143} <= lengths
    assert len(manifest) == len(module.BROAD_SEQ_LENS) * 2
    assert {case.variant_id for case in manifest} == {
        "per_q_s1_w8+unified_s2_w8",
        "grouped_s1_bn16_w2_s2+unified_s2_w8",
    }


def test_tuning_variant_metadata_records_actual_block_and_stages():
    module = _load_module()
    args = SimpleNamespace(
        variants="grouped_s1_bn32_w2_s2,grouped_s1_bn16_w2_s3",
        stages="stage1",
        dtypes="bf16",
        score_modes="none",
        batch_sizes="1",
        seq_lens="262144",
        head_dims="256",
        block_seqs="256",
        slot_orders="ordered",
        num_heads=16,
        num_kv_heads=4,
        seed=20260710,
        warmup=25,
        rounds=5,
        iterations=100,
    )
    by_variant = {case.variant_id: case for case in module.build_manifest(args)}
    assert by_variant["grouped_s1_bn32_w2_s2"].block_n == 32
    assert by_variant["grouped_s1_bn32_w2_s2"].num_stages == 2
    assert by_variant["grouped_s1_bn16_w2_s3"].block_n == 16
    assert by_variant["grouped_s1_bn16_w2_s3"].num_stages == 3


def test_aggregate_keeps_failures_and_computes_baseline_speedup():
    module = _load_module()
    common = {
        "stage": "stage1",
        "head_dim": 256,
        "B": 1,
        "Hq": 16,
        "Hkv": 4,
        "dtype": "bf16",
        "score_mode": "none",
        "block_seq": 256,
        "context_len": 256,
        "slot_case": "ordered",
        "layout_case": "contiguous",
        "required": True,
    }
    rows = [
        {
            **common,
            "case_id": "baseline",
            "variant_id": "per_q_s1_w8",
            "num_warps": 8,
            "status": "success",
            "latency_p50_ms": 2.0,
        },
        {
            **common,
            "case_id": "candidate",
            "variant_id": "grouped_s1_allow256_w4",
            "num_warps": 4,
            "status": "success",
            "latency_p50_ms": 1.0,
        },
        {
            **common,
            "case_id": "failed",
            "variant_id": "grouped_s1_allow256_w8",
            "num_warps": 8,
            "status": "model_failed",
            "reason": "compile failed",
        },
    ]
    aggregate = module.aggregate_results(rows)
    assert aggregate["status_counts"]["model_failed"] == 1
    assert not aggregate["all_required_success"]
    gate = next(item for item in aggregate["performance_gates"] if item["variant_id"] == "grouped_s1_allow256_w4")
    assert gate["geomean_speedup"] == 2.0
    assert gate["passed"]
