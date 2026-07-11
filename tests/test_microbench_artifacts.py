from types import SimpleNamespace

from collections import defaultdict

from benchmark.microbench import (
    _artifact_records,
    _profiler_snapshot,
    _resolved_engine_config,
)


def test_resolved_engine_config_records_backend_and_jsonable_values():
    llm = SimpleNamespace(
        config=SimpleNamespace(
            vllm_sparse_method="deltakv",
            prefill_schedule_policy="long_bs1full_short_batch",
            chunk_prefill_size=4096,
            decode_cuda_graph=True,
            decode_cuda_graph_capture_sampling=False,
            deltakv_sparse_decode_backend="fa2",
            deltakv_triton_materialize_block_tokens=16,
            deltakv_triton_gather_heads_per_program=4,
            deltakv_triton_reconstruct_heads_per_program=2,
            full_layer_kv_quant_bits=4,
            kv_quant_bits=0,
            kv_quant_group_size=64,
            full_attn_layers=(0, 1, 2, 8),
            obs_layer_ids=[2, 8],
        )
    )

    resolved = _resolved_engine_config(llm)

    assert resolved["deltakv_sparse_decode_backend"] == "fa2"
    assert resolved["full_attn_layers"] == [0, 1, 2, 8]
    assert resolved["obs_layer_ids"] == [2, 8]


def test_profiler_snapshot_records_total_calls_and_average():
    profiler = SimpleNamespace(
        times={"prefill_score_pipeline": 0.012},
        counts=defaultdict(int, {"prefill_score_pipeline": 3}),
    )

    snapshot = _profiler_snapshot(profiler)

    assert snapshot == {
        "prefill_score_pipeline": {
            "calls": 3,
            "total_s": 0.012,
            "avg_ms": 4.0,
        }
    }


def test_artifact_records_preserve_generated_outputs():
    args = SimpleNamespace(output_len=2, temperature=0.0, top_p=1.0)
    generated_outputs = [{"seq_id": 7, "token_ids": [11, 13]}]
    steady_outputs = [{"text": "sample", "token_ids": [11, 13]}]

    records = _artifact_records(
        args,
        [
            {
                "status": "SUCCESS",
                "length": 16,
                "generated_outputs": generated_outputs,
                "steady_state_samples": [{"outputs": steady_outputs}],
            }
        ],
    )

    assert records[0]["generated_outputs"] == generated_outputs
    assert records[0]["steady_state_samples"][0]["outputs"] == steady_outputs
