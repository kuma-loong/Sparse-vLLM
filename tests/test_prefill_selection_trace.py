from types import SimpleNamespace

import torch

from sparsevllm.engine.sparse_controller import SparseController


def test_prefill_selection_trace_records_selection_and_cache_state():
    controller = SparseController.__new__(SparseController)
    controller.prefill_selection_trace_enabled = True
    controller.prefill_selection_trace = []
    controller.sparse_method = "snapkv"
    controller.num_sink = 1
    controller.num_recent = 1

    scores = torch.tensor([0.0, 0.1, 0.8, 0.2, 0.0], dtype=torch.float32)
    keep_indices = torch.tensor([0, 2, 4], dtype=torch.long)
    source_slots = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    cache_slots = torch.tensor([10, 12, 14], dtype=torch.int32)
    seq = SimpleNamespace(
        seq_id=7,
        num_prefilled_tokens=3,
        current_chunk_size=2,
    )

    controller._record_prefill_selection_trace(
        layer_idx=4,
        seq=seq,
        kv_len=5,
        budget=3,
        scores=scores,
        keep_indices=keep_indices,
        source_slots=source_slots,
        cache_slots=cache_slots,
        pool_kernel_size=1,
    )

    assert len(controller.prefill_selection_trace) == 1
    trace = controller.prefill_selection_trace[0]
    assert trace["layer_idx"] == 4
    assert trace["selected_token_indices"] == [0, 2, 4]
    assert trace["cache_order_token_indices"] == [0, 2, 4]
    assert trace["selected_source_slots"] == [10, 12, 14]
    assert trace["cache_slots"] == [10, 12, 14]
    assert trace["cutoff_margin"] == torch.tensor(0.6).item()
    assert trace["score_finite"] is True
    assert len(trace["score_sha256"]) == 64
