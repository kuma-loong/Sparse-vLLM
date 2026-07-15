import unittest

import pytest
import torch
import torch.nn.functional as F

from sparsevllm.triton_kernel.moe import fused_moe, moe_align_block_size


def _oracle_local_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    local_expert_start: int,
) -> torch.Tensor:
    output = torch.zeros_like(hidden_states)
    local_expert_end = local_expert_start + int(w13_weight.shape[0])
    for local_expert_id in range(int(w13_weight.shape[0])):
        global_expert_id = local_expert_start + local_expert_id
        token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
        if token_ids.numel() == 0:
            continue
        assert local_expert_start <= global_expert_id < local_expert_end
        gate_up = F.linear(hidden_states[token_ids], w13_weight[local_expert_id])
        gate, up = gate_up.chunk(2, dim=-1)
        expert_output = F.linear(F.silu(gate) * up, w2_weight[local_expert_id])
        expert_output *= topk_weights[token_ids, topk_slots, None]
        output.index_add_(0, token_ids, expert_output.to(output.dtype))
    return output


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_moe_align_block_size_filters_ep_experts_and_pads_blocks():
    topk_ids = torch.tensor(
        [[2, 0], [3, 2], [5, 2], [0, 1]],
        dtype=torch.int64,
        device="cuda",
    )
    sorted_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        4,
        6,
        local_expert_start=2,
        local_expert_end=4,
    )
    torch.cuda.synchronize()

    invalid = topk_ids.numel()
    assert int(num_tokens_post_padded.item()) == 8
    assert expert_ids[:2].tolist() == [0, 1]
    assert all(expert_id == -1 for expert_id in expert_ids[2:].tolist())
    assert sorted(sorted_ids[:4].tolist()) == sorted([0, 3, 5, invalid])
    assert sorted(sorted_ids[4:8].tolist()) == sorted([2, invalid, invalid, invalid])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_naive_decode_matches_oracle(dtype):
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_experts = 16
    hidden_size = 37
    intermediate_size = 23
    hidden_states = torch.randn(1, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.tensor([[3, 12]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor([[0.65, 0.35]], dtype=dtype, device=device)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    tolerance = 3e-2 if dtype == torch.bfloat16 else 1e-2
    assert torch.allclose(actual, expected, atol=tolerance, rtol=tolerance)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_aligned_prefill_matches_oracle_with_padding():
    torch.manual_seed(12)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 19
    num_experts = 7
    top_k = 3
    hidden_size = 45
    intermediate_size = 29
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_ids[:5] = 0
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=0,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=4e-2, rtol=4e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_ep_local_output_matches_oracle_and_ignores_remote_experts():
    torch.manual_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_tokens = 13
    num_experts = 8
    local_expert_start = 2
    num_local_experts = 4
    top_k = 2
    hidden_size = 32
    intermediate_size = 17
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w13_weight = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(num_tokens, top_k, device=device, dtype=dtype)

    expected = _oracle_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        local_expert_start,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=local_expert_start,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_can_preserve_fp32_topk_sum():
    torch.manual_seed(15)
    device = torch.device("cuda")
    hidden_states = torch.randn(5, 31, device=device, dtype=torch.bfloat16)
    w13_weight = torch.randn(4, 38, 31, device=device, dtype=torch.bfloat16)
    w2_weight = torch.randn(4, 31, 19, device=device, dtype=torch.bfloat16)
    topk_ids = torch.tensor(
        [[0, 3], [1, 2], [2, 0], [3, 1], [0, 2]],
        dtype=torch.int64,
        device=device,
    )
    topk_weights = torch.rand(5, 2, device=device, dtype=torch.bfloat16)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=4,
        local_expert_start=0,
        output_dtype=torch.float32,
    )
    torch.cuda.synchronize()

    assert actual.dtype == torch.float32


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for Triton MoE tests.")
def test_triton_moe_returns_zero_when_all_assignments_are_remote():
    torch.manual_seed(14)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden_states = torch.randn(9, 24, device=device, dtype=dtype)
    w13_weight = torch.randn(4, 26, 24, device=device, dtype=dtype)
    w2_weight = torch.randn(4, 24, 13, device=device, dtype=dtype)
    topk_ids = torch.zeros(9, 2, dtype=torch.int64, device=device)
    topk_weights = torch.full((9, 2), 0.5, dtype=dtype, device=device)

    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=8,
        local_expert_start=4,
    )
    torch.cuda.synchronize()

    assert torch.count_nonzero(actual).item() == 0
