"""Real CUDA checks for direct FP8 cache writes and FlashInfer consumption."""

import math
from types import SimpleNamespace

import pytest
import torch

from sparseengine.configs.fp8_kv_scales import FP8KVScales
from sparseengine.engine.cache_manager.storage.quantized_kv import QuantizedKVStorage
from sparseengine.kernels.triton.quantized_kv import materialize_sequence, write_fp8_kv
from sparseengine.operators.decode_attention import (
    DecodeAttentionOpSpec,
    FlashInferPagedDecodeAttentionProvider,
)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires an idle CUDA device")
    return torch.device("cuda")


def _cache(device, *, page_size=32, head_dim=128):
    scales = FP8KVScales((0.02,), (0.01,), "test", "test")
    storage = QuantizedKVStorage(
        format="fp8_kv", bits=8, page_size=page_size,
        num_kv_heads=2, head_dim=head_dim, dtype=torch.bfloat16,
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
    torch.testing.assert_close(actual_k.float(), expected_k, atol=0.02, rtol=0)
    torch.testing.assert_close(actual_v.float(), expected_v, atol=0.01, rtol=0)
    assert storage.raw.numel() == 0
    assert storage.key_scale.numel() == storage.value_scale.numel() == 1


def test_flashinfer_reads_fp8_pages_with_bf16_query(device):
    torch.manual_seed(7)
    _, payload = _cache(device)
    lengths = [31, 65]
    page_ids = ((5, 1, 7), (3, 0, 4))
    slots = torch.zeros(2, 96, device=device, dtype=torch.int32)
    histories = []
    for row, length in enumerate(lengths):
        page_slots = torch.tensor(page_ids[row], device=device, dtype=torch.int32)
        slots[row] = (page_slots[:, None] * 32 + torch.arange(32, device=device)).reshape(-1)
        key = torch.randn(length, 2, 128, device=device, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        write_fp8_kv(key, value, payload, slots[row, :length])
        histories.append((
            payload.k_cache.view(-1, 2, 128)[slots[row, :length].long()].float() * 0.02,
            payload.v_cache.view(-1, 2, 128)[slots[row, :length].long()].float() * 0.01,
        ))
    q = torch.randn(2, 4, 128, device=device, dtype=torch.bfloat16)
    context_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    spec = DecodeAttentionOpSpec(
        num_query_heads=4, num_kv_heads=2, head_dim=128,
        activation_dtype=torch.bfloat16, softmax_scale=128 ** -0.5,
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
        scores = torch.einsum("hd,hnd->hn", q[row].float(), expanded_k) / math.sqrt(128)
        expected = torch.einsum("hn,hnd->hd", scores.softmax(-1), expanded_v)
        torch.testing.assert_close(result[row].float(), expected, atol=0.035, rtol=0.035)
    provider.close()
