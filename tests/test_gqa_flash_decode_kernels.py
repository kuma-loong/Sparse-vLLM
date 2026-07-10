import math
import os

import pytest
import torch

from sparsevllm.engine.cache_manager import DecodeComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend
from sparsevllm.triton_kernel.flash_decoding_stage2 import flash_decode_stage2
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import (
    _PRODUCTION_VARIANT_ID,
    _requires_64bit_cache_offsets,
    flash_decode_stage1,
    flash_decode_stage1_variant,
    flash_decode_stage1_with_score,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernel tests")
SEED = 20260710
SENTINEL = 12345.0


@pytest.fixture(autouse=True)
def _enable_bounds_checks(monkeypatch):
    monkeypatch.setenv("SVLLM_DEBUG_DECODE_BOUNDS", "1")


def _strided_random(shape, *, dtype, stride_factor, device="cuda"):
    storage_shape = (*shape[:-1], shape[-1] * stride_factor)
    storage = torch.randn(storage_shape, dtype=dtype, device=device)
    return storage[..., ::stride_factor]


def _make_case(
    *,
    context_lens,
    hq,
    hkv,
    head_dim,
    dtype,
    max_len,
    block_seq,
    score_mode,
    score_dtype=torch.float32,
    strided=False,
    shuffled=False,
    all_negative_qk=False,
):
    torch.manual_seed(SEED)
    batch = len(context_lens)
    slot_capacity = batch * max_len * 2
    q = (
        _strided_random((batch, hq, head_dim), dtype=dtype, stride_factor=2)
        if strided
        else torch.randn((batch, hq, head_dim), dtype=dtype, device="cuda")
    )
    if strided:
        k_storage = torch.randn((slot_capacity * 2, hkv + 1, head_dim * 2), dtype=dtype, device="cuda")
        v_storage = torch.randn((slot_capacity * 3, hkv + 2, head_dim * 3), dtype=dtype, device="cuda")
        k = k_storage[::2, :hkv, ::2]
        v = v_storage[::3, :hkv, ::3]
    else:
        k = torch.randn((slot_capacity, hkv, head_dim), dtype=dtype, device="cuda")
        v = torch.randn_like(k)
    if all_negative_qk:
        q = q.abs()
        k = -k.abs()

    table_rows = batch + 2
    if strided:
        req_storage = torch.full((table_rows, max_len * 2), -1, dtype=torch.int32, device="cuda")
        req_to_tokens = req_storage[:, ::2]
    else:
        req_to_tokens = torch.full((table_rows, max_len), -1, dtype=torch.int32, device="cuda")
    req_indices = torch.arange(1, batch + 1, dtype=torch.int32, device="cuda")
    if batch > 1:
        req_indices = req_indices.roll(1)
    for batch_idx in range(batch):
        row = int(req_indices[batch_idx].item())
        slots = torch.arange(batch_idx * max_len * 2, (batch_idx + 1) * max_len * 2, 2, device="cuda")
        if shuffled:
            slots = slots[torch.randperm(max_len, device="cuda")]
        if batch_idx > 0 and max_len >= 4:
            slots[:4] = req_to_tokens[int(req_indices[0].item()), :4]
        req_to_tokens[row, :] = slots.to(torch.int32)
    b_seqlen = torch.tensor(context_lens, dtype=torch.int32, device="cuda")

    blocks = math.ceil(max_len / block_seq)
    if strided:
        mid_storage = torch.full(
            (batch, hq, blocks + 1, head_dim * 2), SENTINEL, dtype=torch.float32, device="cuda"
        )
        lse_storage = torch.full((batch, hq, (blocks + 1) * 2), SENTINEL, dtype=torch.float32, device="cuda")
        out_storage = torch.full((batch, hq, head_dim * 2), SENTINEL, dtype=dtype, device="cuda")
        mid_o = mid_storage[:, :, :blocks, ::2]
        mid_lse = lse_storage[:, :, : blocks * 2 : 2]
        output = out_storage[..., ::2]
    else:
        mid_o = torch.full((batch, hq, blocks, head_dim), SENTINEL, dtype=torch.float32, device="cuda")
        mid_lse = torch.full((batch, hq, blocks), SENTINEL, dtype=torch.float32, device="cuda")
        output = torch.empty((batch, hq, head_dim), dtype=dtype, device="cuda")

    score = None
    if score_mode == "3d":
        if strided:
            score_storage = torch.full(
                (batch, hq, max_len * 2), -float("inf"), dtype=score_dtype, device="cuda"
            )
            score = score_storage[..., ::2]
        else:
            score = torch.full((batch, hq, max_len), -float("inf"), dtype=score_dtype, device="cuda")
    elif score_mode == "2d":
        if strided:
            score_storage = torch.full((batch, max_len * 2), -float("inf"), dtype=score_dtype, device="cuda")
            score = score_storage[:, ::2]
        else:
            score = torch.full((batch, max_len), -float("inf"), dtype=score_dtype, device="cuda")

    return {
        "q": q,
        "k": k,
        "v": v,
        "req_to_tokens": req_to_tokens,
        "req_indices": req_indices,
        "context_lens": b_seqlen,
        "max_len": max_len,
        "block_seq": block_seq,
        "mid_o": mid_o,
        "mid_lse": mid_lse,
        "output": output,
        "score": score,
        "score_mode": score_mode,
    }


def _oracle(case):
    q, k, v = case["q"], case["k"], case["v"]
    hq, hkv, head_dim = q.shape[1], k.shape[1], q.shape[2]
    group_size = hq // hkv
    kv_head_for_q = torch.arange(hq, device="cuda") // group_size
    final_rows = []
    mid_o_rows = []
    mid_lse_rows = []
    score_rows = []
    for batch_idx, context_len in enumerate(case["context_lens"].tolist()):
        req_row = int(case["req_indices"][batch_idx].item())
        slots = case["req_to_tokens"][req_row, :context_len].long()
        key = k.index_select(0, slots)[:, kv_head_for_q, :].float().transpose(0, 1)
        value = v.index_select(0, slots)[:, kv_head_for_q, :].float().transpose(0, 1)
        raw_qk = torch.sum(q[batch_idx].float()[:, None, :] * key, dim=-1)
        logits = raw_qk / math.sqrt(head_dim)
        final_rows.append(torch.sum(torch.softmax(logits, dim=-1)[..., None] * value, dim=1))
        block_o, block_lse = [], []
        for start in range(0, context_len, case["block_seq"]):
            end = min(start + case["block_seq"], context_len)
            block_logits = logits[:, start:end]
            block_o.append(
                torch.sum(torch.softmax(block_logits, dim=-1)[..., None] * value[:, start:end], dim=1)
            )
            block_lse.append(torch.logsumexp(block_logits, dim=-1))
        mid_o_rows.append(block_o)
        mid_lse_rows.append(block_lse)
        score_rows.append(raw_qk)
    return torch.stack(final_rows), mid_o_rows, mid_lse_rows, score_rows


def _assert_case(case, variant_id=None):
    if variant_id is not None:
        flash_decode_stage1_variant(
            case["q"],
            case["k"],
            case["v"],
            case["req_to_tokens"],
            case["req_indices"],
            case["context_lens"],
            case["max_len"],
            case["mid_o"],
            case["mid_lse"],
            case["block_seq"],
            variant_id=variant_id,
            attn_score=case["score"],
        )
    elif case["score"] is None:
        flash_decode_stage1(
            case["q"],
            case["k"],
            case["v"],
            case["req_to_tokens"],
            case["req_indices"],
            case["context_lens"],
            case["max_len"],
            case["mid_o"],
            case["mid_lse"],
            case["block_seq"],
        )
    else:
        flash_decode_stage1_with_score(
            case["q"],
            case["k"],
            case["v"],
            case["req_to_tokens"],
            case["req_indices"],
            case["context_lens"],
            case["max_len"],
            case["mid_o"],
            case["mid_lse"],
            case["score"],
            case["block_seq"],
        )
    flash_decode_stage2(
        case["mid_o"],
        case["mid_lse"],
        case["context_lens"],
        case["output"],
        case["block_seq"],
    )
    torch.cuda.synchronize()
    final_ref, mid_o_ref, mid_lse_ref, score_ref = _oracle(case)
    tolerance = 5e-3 if case["q"].dtype == torch.float16 else 2e-2
    torch.testing.assert_close(case["output"].float(), final_ref, rtol=tolerance, atol=tolerance)
    assert torch.isfinite(case["output"]).all()
    for batch_idx, context_len in enumerate(case["context_lens"].tolist()):
        valid_blocks = math.ceil(context_len / case["block_seq"])
        for block_idx in range(valid_blocks):
            torch.testing.assert_close(
                case["mid_o"][batch_idx, :, block_idx],
                mid_o_ref[batch_idx][block_idx],
                rtol=tolerance,
                atol=tolerance,
            )
            torch.testing.assert_close(
                case["mid_lse"][batch_idx, :, block_idx],
                mid_lse_ref[batch_idx][block_idx],
                rtol=tolerance,
                atol=tolerance,
            )
        if valid_blocks < case["mid_o"].shape[2]:
            assert torch.equal(
                case["mid_o"][batch_idx, :, valid_blocks:],
                torch.full_like(case["mid_o"][batch_idx, :, valid_blocks:], SENTINEL),
            )
            assert torch.equal(
                case["mid_lse"][batch_idx, :, valid_blocks:],
                torch.full_like(case["mid_lse"][batch_idx, :, valid_blocks:], SENTINEL),
            )
        if case["score_mode"] == "3d":
            torch.testing.assert_close(
                case["score"][batch_idx, :, :context_len].float(),
                score_ref[batch_idx],
                rtol=tolerance,
                atol=tolerance,
            )
            assert torch.all(case["score"][batch_idx, :, context_len:].float() < -1e10)
        elif case["score_mode"] == "2d":
            torch.testing.assert_close(
                case["score"][batch_idx, :context_len].float(),
                score_ref[batch_idx].max(dim=0).values,
                rtol=tolerance,
                atol=tolerance,
            )
            assert torch.all(case["score"][batch_idx, context_len:].float() < -1e10)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("score_mode", ["none", "3d", "2d"])
def test_d256_ragged_length_boundaries(dtype, score_mode):
    case = _make_case(
        context_lens=[1, 15, 16, 17, 255, 256, 257, 513],
        hq=16,
        hkv=4,
        head_dim=256,
        dtype=dtype,
        max_len=1024,
        block_seq=256,
        score_mode=score_mode,
        all_negative_qk=score_mode == "2d",
    )
    _assert_case(case, variant_id="grouped_s1_allow256_w2")


def test_production_variant_is_shape_independent():
    assert _PRODUCTION_VARIANT_ID == "grouped_s1_bn16_w2_s2"


def test_cache_offset_width_is_derived_from_tensor_geometry():
    class TensorGeometry:
        def __init__(self, shape, stride):
            self.shape = shape
            self._stride = stride

        def stride(self):
            return self._stride

    int32_safe = TensorGeometry((2097152, 4, 256), (1024, 256, 1))
    needs_int64 = TensorGeometry((2097153, 4, 256), (1024, 256, 1))
    assert not _requires_64bit_cache_offsets(int32_safe)
    assert _requires_64bit_cache_offsets(needs_int64)


@pytest.mark.parametrize("block_seq", [128, 512])
def test_d256_block_seq_boundaries(block_seq):
    case = _make_case(
        context_lens=[127, 128, 129, 511, 512, 513],
        hq=16,
        hkv=4,
        head_dim=256,
        dtype=torch.bfloat16,
        max_len=1024,
        block_seq=block_seq,
        score_mode="none",
    )
    _assert_case(case)


@pytest.mark.parametrize(
    "variant_id",
    [
        "grouped_s1_bn16_w2_s1",
        "grouped_s1_bn16_w2_s3",
        "grouped_s1_bn32_w2_s2",
        "grouped_s1_bn64_w2_s2",
        "grouped_s1_bn128_w2_s2",
    ],
)
def test_d256_tuning_variants_match_fp32_oracle(variant_id):
    case = _make_case(
        context_lens=[17, 63, 129, 257],
        hq=16,
        hkv=4,
        head_dim=256,
        dtype=torch.bfloat16,
        max_len=512,
        block_seq=256,
        score_mode="none",
    )
    _assert_case(case, variant_id=variant_id)


@pytest.mark.skipif(
    os.environ.get("SVLLM_RUN_LONG_GPU_TESTS") != "1",
    reason="set SVLLM_RUN_LONG_GPU_TESTS=1 for the 256K baseline/candidate cross-check",
)
def test_d256_256k_baseline_matches_grouped_output():
    torch.manual_seed(SEED)
    batch, hq, hkv, head_dim, context_len, block_seq = 9, 16, 4, 256, 262144, 256
    q = torch.randn((batch, hq, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch * context_len, hkv, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    slots = torch.arange(batch * context_len, dtype=torch.int32, device="cuda").view(batch, context_len)
    req_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    context_lens = torch.full((batch,), context_len, dtype=torch.int32, device="cuda")
    blocks = math.ceil(context_len / block_seq)

    outputs = {}
    for variant_id in ("per_q_s1_w8", "grouped_s1_allow256_w2"):
        mid_o = torch.empty((batch, hq, blocks, head_dim), dtype=torch.float32, device="cuda")
        mid_lse = torch.empty((batch, hq, blocks), dtype=torch.float32, device="cuda")
        output = torch.empty_like(q)
        flash_decode_stage1_variant(
            q,
            k,
            v,
            slots,
            req_indices,
            context_lens,
            context_len,
            mid_o,
            mid_lse,
            block_seq,
            variant_id=variant_id,
        )
        flash_decode_stage2(mid_o, mid_lse, context_lens, output, block_seq)
        torch.cuda.synchronize()
        assert torch.isfinite(mid_o).all()
        assert torch.isfinite(mid_lse).all()
        assert torch.isfinite(output).all()
        outputs[variant_id] = output.float()
    torch.testing.assert_close(
        outputs["grouped_s1_allow256_w2"],
        outputs["per_q_s1_w8"],
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.parametrize("head_dim", [16, 17, 24, 31, 40, 65, 80, 127, 160, 192, 224, 255, 256])
def test_grouped_regression_head_dims(head_dim):
    case = _make_case(
        context_lens=[17, 129],
        hq=16,
        hkv=4,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        max_len=256,
        block_seq=128,
        score_mode="none",
    )
    _assert_case(case)


@pytest.mark.parametrize("hq,hkv", [(8, 4), (12, 4), (16, 4), (28, 4), (32, 4), (32, 2)])
def test_d256_gqa_ratios(hq, hkv):
    case = _make_case(
        context_lens=[17],
        hq=hq,
        hkv=hkv,
        head_dim=256,
        dtype=torch.bfloat16,
        max_len=32,
        block_seq=256,
        score_mode="none",
    )
    _assert_case(case, variant_id="grouped_s1_allow256_w2")


@pytest.mark.parametrize("score_mode,score_dtype", [("3d", torch.float16), ("3d", torch.bfloat16)])
def test_d256_strided_layout_and_score_dtype(score_mode, score_dtype):
    case = _make_case(
        context_lens=[17, 257],
        hq=16,
        hkv=4,
        head_dim=256,
        dtype=torch.bfloat16,
        max_len=512,
        block_seq=256,
        score_mode=score_mode,
        score_dtype=score_dtype,
        strided=True,
        shuffled=True,
        all_negative_qk=score_mode == "2d",
    )
    _assert_case(case)


def test_invalid_inputs_fail_before_launch():
    case = _make_case(
        context_lens=[17],
        hq=16,
        hkv=4,
        head_dim=256,
        dtype=torch.bfloat16,
        max_len=32,
        block_seq=256,
        score_mode="none",
    )
    args = [
        case["q"],
        case["k"],
        case["v"],
        case["req_to_tokens"],
        case["req_indices"],
        case["context_lens"],
        case["max_len"],
        case["mid_o"],
        case["mid_lse"],
        case["block_seq"],
    ]
    bad = list(args)
    bad[1] = case["k"][..., :128]
    with pytest.raises(ValueError, match="head dimensions must match"):
        flash_decode_stage1(*bad)
    bad = list(args)
    bad[9] = 15
    with pytest.raises(ValueError, match="block_seq"):
        flash_decode_stage1(*bad)
    bad = list(args)
    bad[6] = 0
    with pytest.raises(ValueError, match="max_len_in_batch"):
        flash_decode_stage1(*bad)
    bad = list(args)
    bad[7] = case["mid_o"][:, :, :0]
    with pytest.raises(ValueError, match="mid_out capacity"):
        flash_decode_stage1(*bad)
    bad_slots = case["req_to_tokens"].clone()
    bad_slots[int(case["req_indices"][0].item()), 0] = case["k"].shape[0]
    bad = list(args)
    bad[3] = bad_slots
    with pytest.raises(ValueError, match="slot index out of range"):
        flash_decode_stage1(*bad)


@pytest.mark.parametrize("with_score", [False, True])
def test_backend_eager_decode_for_32_steps(with_score):
    torch.manual_seed(SEED)
    hq, hkv, head_dim, max_len, block_seq = 16, 4, 256, 32, 256
    q = torch.randn((1, hq, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((max_len, hkv, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    slots = torch.arange(max_len, dtype=torch.int32, device="cuda").view(1, max_len)
    req_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    mid_o = torch.empty((1, hq, 1, head_dim), dtype=torch.float32, device="cuda")
    mid_lse = torch.empty((1, hq, 1), dtype=torch.float32, device="cuda")
    backend = TritonAttentionBackend()
    kv_head_for_q = torch.arange(hq, device="cuda") // (hq // hkv)
    for step in range(1, 33):
        context_lens = torch.tensor([step], dtype=torch.int32, device="cuda")
        score = (
            torch.full((1, hq, max_len), -float("inf"), dtype=torch.float32, device="cuda")
            if with_score
            else None
        )
        view = DecodeComputeView(
            k_cache=k,
            v_cache=v,
            active_slots=slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=score,
            max_context_len=step,
        )
        output = backend.run_decode(
            q,
            view,
            mid_o=mid_o,
            mid_o_logexpsum=mid_lse,
            max_len_in_batch=max_len,
            block_seq=block_seq,
            num_heads=hq,
            num_kv_heads=hkv,
        )
        key = k[:step, kv_head_for_q].float().transpose(0, 1)
        value = v[:step, kv_head_for_q].float().transpose(0, 1)
        logits = torch.sum(q[0].float()[:, None, :] * key, dim=-1) / math.sqrt(head_dim)
        reference = torch.sum(torch.softmax(logits, dim=-1)[..., None] * value, dim=1)
        torch.testing.assert_close(output[0].float(), reference, rtol=2e-2, atol=2e-2)
        if score is not None:
            raw_qk = torch.sum(q[0].float()[:, None, :] * key, dim=-1)
            torch.testing.assert_close(score[0, :, :step], raw_qk, rtol=2e-2, atol=2e-2)


def test_backend_cuda_graph_capture_and_replay():
    torch.manual_seed(SEED)
    hq, hkv, head_dim, context_len, block_seq = 16, 4, 256, 32, 256
    q = torch.randn((1, hq, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((context_len, hkv, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)
    slots = torch.arange(context_len, dtype=torch.int32, device="cuda").view(1, context_len)
    req_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    context_lens = torch.full((1,), context_len, dtype=torch.int32, device="cuda")
    view = DecodeComputeView(
        k_cache=k,
        v_cache=v,
        active_slots=slots,
        req_indices=req_indices,
        context_lens=context_lens,
        attn_score=None,
        max_context_len=context_len,
    )
    mid_o = torch.empty((1, hq, 1, head_dim), dtype=torch.float32, device="cuda")
    mid_lse = torch.empty((1, hq, 1), dtype=torch.float32, device="cuda")
    backend = TritonAttentionBackend()

    def decode():
        return backend.run_decode(
            q,
            view,
            mid_o=mid_o,
            mid_o_logexpsum=mid_lse,
            max_len_in_batch=context_len,
            block_seq=block_seq,
            num_heads=hq,
            num_kv_heads=hkv,
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            decode()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = decode()
    q.copy_(torch.randn_like(q))
    graph.replay()
    torch.cuda.synchronize()
    kv_head_for_q = torch.arange(hq, device="cuda") // (hq // hkv)
    key = k[:, kv_head_for_q].float().transpose(0, 1)
    value = v[:, kv_head_for_q].float().transpose(0, 1)
    logits = torch.sum(q[0].float()[:, None, :] * key, dim=-1) / math.sqrt(head_dim)
    reference = torch.sum(torch.softmax(logits, dim=-1)[..., None] * value, dim=1)
    torch.testing.assert_close(graph_output[0].float(), reference, rtol=2e-2, atol=2e-2)
