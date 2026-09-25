"""Real CUDA checks for direct FP8 cache writes and FlashInfer consumption."""

import math
from types import SimpleNamespace

import pytest
import torch

from sparseengine.configs.fp8_kv_scales import FP8KVScales
from sparseengine.engine.decode_graph_contract import DecodeGraphContract, DecodeGraphInputs
from sparseengine.engine.cache_manager.storage.quantized_kv import QuantizedKVStorage
from sparseengine.engine.cache_manager.base import ExplicitKVWrite
from sparseengine.kernels.triton.quantized_kv import materialize_sequence, write_fp8_kv
from sparseengine.operators.decode_attention import (
    DecodeAttentionOpSpec,
    FlashInferPagedDecodeAttentionProvider,
)
from sparseengine.operators.prefill_attention import (
    FlashInferFp8Fa2PagedPrefillAttentionProvider,
    PrefillAttentionOpSpec,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires an idle CUDA device")
    return torch.device("cuda")


def _cache(device, *, page_size=32, head_dim=128, activation_dtype=torch.bfloat16):
    scales = FP8KVScales((0.02,), (0.01,), "test", "test")
    storage = QuantizedKVStorage(
        format="fp8_kv", bits=8, page_size=page_size,
        num_kv_heads=2, head_dim=head_dim, dtype=activation_dtype,
        seed=0, fp8_scales=scales,
    )
    storage.allocate(num_layers=1, num_slots=8 * page_size, num_rows=2, device=device)
    return storage, storage.layer_payload(0)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 65])
def test_static_fp8_write_and_history_have_no_bf16_tail(device, length):
    torch.manual_seed(10 + length)
    storage, payload = _cache(device)
    # The first request owns nonadjacent physical pages.
    pages = torch.tensor([5, 1, 7], device=device, dtype=torch.int32)
    slots = (pages[:, None] * 32 + torch.arange(32, device=device)).reshape(-1).int()
    key = torch.randn(length, 2, 128, device=device, dtype=torch.bfloat16) * 2
    value = torch.randn_like(key)
    write_fp8_kv(key, value, payload, slots[:length])
    expected_k = (key.float() / 0.02).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.02
    expected_v = (value.float() / 0.01).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.01
    actual_k = torch.empty_like(key)
    actual_v = torch.empty_like(value)
    table = slots[None]
    materialize_sequence(payload, table, 0, length, actual_k, actual_v)
    torch.testing.assert_close(actual_k.float(), expected_k.bfloat16().float(), atol=0.02, rtol=0)
    torch.testing.assert_close(actual_v.float(), expected_v.bfloat16().float(), atol=0.02, rtol=0)
    assert storage.raw.numel() == 0
    assert storage.key_scale.numel() == storage.value_scale.numel() == 1


@pytest.mark.parametrize(
    "head_dim,activation_dtype",
    [(64, torch.bfloat16), (128, torch.bfloat16), (256, torch.bfloat16),
     (128, torch.float16)],
)
def test_flashinfer_reads_fp8_pages_with_half_precision_query(device, head_dim, activation_dtype):
    torch.manual_seed(7)
    _, payload = _cache(device, head_dim=head_dim, activation_dtype=activation_dtype)
    lengths = [31, 65]
    page_ids = ((5, 1, 7), (3, 0, 4))
    slots = torch.zeros(2, 96, device=device, dtype=torch.int32)
    histories = []
    for row, length in enumerate(lengths):
        page_slots = torch.tensor(page_ids[row], device=device, dtype=torch.int32)
        slots[row] = (page_slots[:, None] * 32 + torch.arange(32, device=device)).reshape(-1)
        key = torch.randn(length, 2, head_dim, device=device, dtype=activation_dtype)
        value = torch.randn_like(key)
        write_fp8_kv(key, value, payload, slots[row, :length])
        histories.append((
            payload.k_cache.view(-1, 2, head_dim)[slots[row, :length].long()].float() * 0.02,
            payload.v_cache.view(-1, 2, head_dim)[slots[row, :length].long()].float() * 0.01,
        ))
    q = torch.randn(2, 4, head_dim, device=device, dtype=activation_dtype)
    context_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    spec = DecodeAttentionOpSpec(
        num_query_heads=4, num_kv_heads=2, head_dim=head_dim,
        activation_dtype=activation_dtype, softmax_scale=head_dim ** -0.5,
        max_batch_size=2, page_size=32, context_capacity=96,
        cuda_graph=False, kv_storage_format="fp8_kv",
    )
    provider = FlashInferPagedDecodeAttentionProvider()
    provider.prepare(spec, device_index=device.index)
    view = SimpleNamespace(
        payload=payload,
        meta=SimpleNamespace(
            active_slots=slots,
            req_indices=torch.arange(2, device=device, dtype=torch.int32),
            context_lens=context_lens,
            max_context_len=max(lengths),
        ),
    )
    result = provider.run(spec, q, view)
    for row, (key, value) in enumerate(histories):
        expanded_k = key.repeat_interleave(2, 1).transpose(0, 1)
        expanded_v = value.repeat_interleave(2, 1).transpose(0, 1)
        scores = torch.einsum("hd,hnd->hn", q[row].float(), expanded_k) / math.sqrt(head_dim)
        expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), expanded_v)
        torch.testing.assert_close(result[row].float(), expected, atol=0.035, rtol=0.035)
    provider.close()


def test_flashinfer_fp8_graph_replans_lengths_and_reads_new_pages(device):
    torch.manual_seed(18)
    _, payload = _cache(device)
    page_size = 32
    slots = torch.zeros(2, 96, device=device, dtype=torch.int32)
    for row, pages in enumerate(((5, 1, 7), (3, 0, 4))):
        slots[row] = (
            torch.tensor(pages, device=device, dtype=torch.int32)[:, None] * page_size
            + torch.arange(page_size, device=device)
        ).reshape(-1)
        key = torch.randn(96, 2, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        write_fp8_kv(key, value, payload, slots[row])
    contract = DecodeGraphContract(
        method="fp8_kv", topology_path_id="dense", batch_capacity=2,
        context_capacity=96,
    )
    inputs = DecodeGraphInputs.allocate(contract, device=device, pin_memory=False)
    spec = DecodeAttentionOpSpec(
        num_query_heads=4, num_kv_heads=2, head_dim=128,
        activation_dtype=torch.bfloat16, softmax_scale=128 ** -0.5,
        max_batch_size=2, page_size=32, context_capacity=96,
        cuda_graph=True, kv_storage_format="fp8_kv",
    )
    provider = FlashInferPagedDecodeAttentionProvider()
    provider.prepare(spec, device_index=device.index)
    state = provider.init_decode_graph_state(spec, contract, inputs)
    q = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    view = SimpleNamespace(
        payload=payload,
        meta=SimpleNamespace(active_slots=slots, req_indices=inputs.request_indices,
                             context_lens=inputs.context_lens, attn_score=None),
    )

    def update(lengths, rows):
        inputs.host.context_lens.copy_(torch.tensor(lengths, dtype=torch.int32))
        inputs.host.request_indices.copy_(torch.tensor(rows, dtype=torch.int32))
        inputs.context_lens.copy_(inputs.host.context_lens)
        inputs.request_indices.copy_(inputs.host.request_indices)
        provider.prepare_decode_graph_out(state)
        provider.prepare_decode_graph_in(state)

    try:
        update([31, 65], [0, 1])
        provider.run(spec, q, view)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = provider.run(spec, q, view)
        for lengths, rows in (([31, 65], [0, 1]), ([64, 32], [1, 0]), ([95, 1], [0, 1])):
            update(lengths, rows)
            q.normal_()
            graph.replay()
            for batch, (length, row) in enumerate(zip(lengths, rows)):
                active = slots[row, :length].long()
                key = payload.k_cache.view(-1, 2, 128)[active].float() * 0.02
                value = payload.v_cache.view(-1, 2, 128)[active].float() * 0.01
                key = key.repeat_interleave(2, 1).transpose(0, 1)
                value = value.repeat_interleave(2, 1).transpose(0, 1)
                scores = torch.einsum("hd,hnd->hn", q[batch].float(), key) / math.sqrt(128)
                expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), value)
                torch.testing.assert_close(result[batch].float(), expected, atol=0.035, rtol=0.035)
    finally:
        provider.close_decode_graph_state(state)
        provider.close()


def test_flashinfer_fp8_paged_prefill_matches_causal_reference(device):
    torch.manual_seed(29)
    _, payload = _cache(device)
    lengths = (35, 66)
    chunks = (3, 2)
    page_ids = ((5, 1, 7), (3, 0, 4))
    slots = torch.zeros(2, 96, device=device, dtype=torch.int32)
    for row, pages in enumerate(page_ids):
        slots[row] = (
            torch.tensor(pages, device=device, dtype=torch.int32)[:, None] * 32
            + torch.arange(32, device=device)
        ).reshape(-1)
        key = torch.randn(lengths[row], 2, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        write_fp8_kv(key, value, payload, slots[row, :lengths[row]])
    q = torch.randn(sum(chunks), 4, 128, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor([0, chunks[0], sum(chunks)], device=device, dtype=torch.int32)
    spec = PrefillAttentionOpSpec(
        num_query_heads=4, num_kv_heads=2, head_dim=128,
        activation_dtype=torch.bfloat16, softmax_scale=128 ** -0.5,
        page_size=32, kv_storage_format="fp8_kv",
    )
    view = SimpleNamespace(
        payload=payload,
        meta=SimpleNamespace(active_slots=slots,
                             req_indices=torch.arange(2, device=device, dtype=torch.int32),
                             context_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
                             attn_score=None),
    )
    provider = FlashInferFp8Fa2PagedPrefillAttentionProvider()
    provider.prepare(spec, device_index=device.index)
    try:
        actual = provider.run(
            spec, q, view, qo_indptr=qo_indptr,
            chunk_lens=torch.tensor(chunks, device=device, dtype=torch.int32),
            max_context_len=max(lengths), layer_idx=0,
        )
        for row, (length, chunk) in enumerate(zip(lengths, chunks)):
            active = slots[row, :length].long()
            key = payload.k_cache.view(-1, 2, 128)[active].float() * 0.02
            value = payload.v_cache.view(-1, 2, 128)[active].float() * 0.01
            key = key.repeat_interleave(2, 1).transpose(0, 1)
            value = value.repeat_interleave(2, 1).transpose(0, 1)
            for index in range(chunk):
                query = q[int(qo_indptr[row]) + index].float()
                visible = length - chunk + index + 1
                scores = torch.einsum("hd,hnd->hn", query, key[:, :visible]) / math.sqrt(128)
                expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), value[:, :visible])
                torch.testing.assert_close(
                    actual[int(qo_indptr[row]) + index].float(), expected,
                    atol=0.035, rtol=0.035,
                )
    finally:
        provider.close()


def test_fp8_first_prefill_uses_current_bf16_without_readback(device):
    torch.manual_seed(41)
    _, payload = _cache(device)
    lengths = (33, 65)
    total = sum(lengths)
    key = torch.randn(total, 2, 128, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    q = torch.randn(total, 4, 128, device=device, dtype=torch.bfloat16)
    qo_indptr = torch.tensor([0, lengths[0], total], device=device, dtype=torch.int32)
    spec = PrefillAttentionOpSpec(
        num_query_heads=4, num_kv_heads=2, head_dim=128,
        activation_dtype=torch.bfloat16, softmax_scale=128 ** -0.5,
        page_size=32, kv_storage_format="fp8_kv",
    )
    view = SimpleNamespace(current_kv=ExplicitKVWrite(key, value))
    provider = FlashInferFp8Fa2PagedPrefillAttentionProvider()
    provider.prepare(spec, device_index=device.index)
    try:
        actual = provider.run(
            spec, q, view, qo_indptr=qo_indptr,
            chunk_lens=torch.tensor(lengths, device=device, dtype=torch.int32),
            max_context_len=max(lengths), layer_idx=0,
        )
        for row, length in enumerate(lengths):
            start = int(qo_indptr[row])
            expanded_k = key[start:start + length].float().repeat_interleave(2, 1)
            expanded_v = value[start:start + length].float().repeat_interleave(2, 1)
            scores = torch.einsum("thd,shd->hts", q[start:start + length].float(), expanded_k)
            scores /= math.sqrt(128)
            scores.masked_fill_(
                torch.ones(length, length, device=device, dtype=torch.bool).triu(1)[None],
                -float("inf"),
            )
            expected = torch.einsum("hts,shd->thd", scores.softmax(-1), expanded_v)
            torch.testing.assert_close(
                actual[start:start + length].float(), expected,
                atol=0.035, rtol=0.035,
            )
    finally:
        provider.close()
