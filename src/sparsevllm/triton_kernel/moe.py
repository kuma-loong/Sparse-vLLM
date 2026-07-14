from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sparsevllm.triton_kernel.silu_and_mul import silu_and_mul_fwd


_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)


@dataclass(frozen=True)
class MoeAlignment:
    sorted_token_ids: torch.Tensor | None
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    block_size: int
    naive: bool


@triton.jit
def _count_local_assignments_kernel(
    topk_ids_ptr,
    counts_ptr,
    num_assignments,
    local_expert_start,
    local_expert_end,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < num_assignments
    global_expert_ids = tl.load(topk_ids_ptr + offsets, mask=valid, other=-1)
    is_local = (
        valid
        & (global_expert_ids >= local_expert_start)
        & (global_expert_ids < local_expert_end)
    )
    local_expert_ids = global_expert_ids - local_expert_start
    tl.atomic_add(counts_ptr + local_expert_ids, 1, mask=is_local)


@triton.jit
def _fill_local_assignments_kernel(
    topk_ids_ptr,
    write_positions_ptr,
    sorted_token_ids_ptr,
    num_assignments,
    local_expert_start,
    local_expert_end,
    BLOCK_SIZE: tl.constexpr,
):
    assignment_ids = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = assignment_ids < num_assignments
    global_expert_ids = tl.load(
        topk_ids_ptr + assignment_ids,
        mask=valid,
        other=-1,
    )
    is_local = (
        valid
        & (global_expert_ids >= local_expert_start)
        & (global_expert_ids < local_expert_end)
    )
    local_expert_ids = global_expert_ids - local_expert_start
    positions = tl.atomic_add(
        write_positions_ptr + local_expert_ids,
        1,
        mask=is_local,
    )
    tl.store(sorted_token_ids_ptr + positions, assignment_ids, mask=is_local)


def _validate_alignment_inputs(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> None:
    if not topk_ids.is_cuda:
        raise ValueError("moe_align_block_size requires CUDA topk_ids.")
    if topk_ids.ndim != 2:
        raise ValueError(
            f"topk_ids must have shape [tokens, top_k], got {tuple(topk_ids.shape)}."
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"topk_ids must use int32 or int64, got dtype={topk_ids.dtype}."
        )
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous.")
    if topk_ids.numel() == 0:
        raise ValueError("topk_ids must contain at least one assignment.")
    if int(block_size) <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if int(num_experts) <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}.")
    if not 0 <= int(local_expert_start) < int(local_expert_end) <= int(num_experts):
        raise ValueError(
            "Invalid local expert range: "
            f"[{local_expert_start}, {local_expert_end}) for num_experts={num_experts}."
        )
    if topk_ids.numel() >= torch.iinfo(torch.int32).max:
        raise ValueError(
            "The Triton MoE assignment representation requires fewer than "
            f"{torch.iinfo(torch.int32).max} assignments, got {topk_ids.numel()}."
        )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    local_expert_start: int = 0,
    local_expert_end: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group local token/expert assignments into padded expert-homogeneous blocks.

    ``sorted_token_ids`` stores indices into flattened ``topk_ids``. Padding uses
    ``topk_ids.numel()`` as an invalid assignment. ``expert_ids`` contains local
    expert indices, not global expert indices. Remote EP assignments are omitted.
    """

    block_size = int(block_size)
    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    if local_expert_end is None:
        local_expert_end = num_experts
    local_expert_end = int(local_expert_end)
    _validate_alignment_inputs(
        topk_ids,
        block_size,
        num_experts,
        local_expert_start,
        local_expert_end,
    )

    num_assignments = int(topk_ids.numel())
    num_local_experts = local_expert_end - local_expert_start
    counts = torch.zeros(
        num_local_experts,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    count_block_size = 256
    _count_local_assignments_kernel[
        (triton.cdiv(num_assignments, count_block_size),)
    ](
        topk_ids,
        counts,
        num_assignments,
        local_expert_start,
        local_expert_end,
        BLOCK_SIZE=count_block_size,
    )

    padded_counts = torch.div(
        counts + block_size - 1,
        block_size,
        rounding_mode="floor",
    ) * block_size
    padded_ends = torch.cumsum(padded_counts, dim=0, dtype=torch.int32)
    expert_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=topk_ids.device),
            padded_ends,
        )
    )

    max_num_tokens_padded = num_assignments + num_local_experts * (block_size - 1)
    max_num_tokens_padded = triton.cdiv(max_num_tokens_padded, block_size) * block_size
    sorted_token_ids = torch.full(
        (max_num_tokens_padded,),
        num_assignments,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    write_positions = expert_offsets[:-1].clone()
    _fill_local_assignments_kernel[
        (triton.cdiv(num_assignments, count_block_size),)
    ](
        topk_ids,
        write_positions,
        sorted_token_ids,
        num_assignments,
        local_expert_start,
        local_expert_end,
        BLOCK_SIZE=count_block_size,
    )

    block_starts = torch.arange(
        max_num_tokens_padded // block_size,
        dtype=torch.int32,
        device=topk_ids.device,
    ) * block_size
    expert_ids = torch.searchsorted(padded_ends, block_starts, right=True).to(
        torch.int32
    )
    num_tokens_post_padded = padded_ends[-1:].contiguous()
    expert_ids = torch.where(
        block_starts < num_tokens_post_padded,
        expert_ids,
        torch.full_like(expert_ids, -1),
    )
    return sorted_token_ids, expert_ids, num_tokens_post_padded


def _gemm_launch_config(num_assignments: int, output_size: int) -> dict[str, int]:
    block_m = 16 if num_assignments <= 256 else 32
    block_n = 64 if output_size <= 4096 else 128
    return {
        "BLOCK_SIZE_M": block_m,
        "BLOCK_SIZE_N": block_n,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 8,
        "num_warps": 4 if block_m == 16 and block_n == 64 else 8,
        "num_stages": 3,
    }


def _prepare_expert_assignment(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    num_assignments = int(topk_ids.numel())
    if num_assignments * 4 <= int(num_experts):
        flat_ids = topk_ids.view(-1)
        is_local = (flat_ids >= local_expert_start) & (
            flat_ids < local_expert_end
        )
        expert_ids = torch.where(
            is_local,
            flat_ids - local_expert_start,
            torch.full_like(flat_ids, -1),
        ).to(torch.int32)
        return MoeAlignment(
            sorted_token_ids=None,
            expert_ids=expert_ids.contiguous(),
            num_tokens_post_padded=torch.full(
                (1,),
                num_assignments * block_size,
                dtype=torch.int32,
                device=topk_ids.device,
            ),
            block_size=block_size,
            naive=True,
        )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        block_size,
        num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )
    return MoeAlignment(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size=block_size,
        naive=False,
    )


@triton.jit
def _routed_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    routing_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_assignments,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    INPUT_TOP_K: tl.constexpr,
    MUL_ROUTING_WEIGHT: tl.constexpr,
    NAIVE_ASSIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    row_offsets = tl.arange(0, BLOCK_SIZE_M)
    if NAIVE_ASSIGNMENT:
        assignment_ids = tl.where(
            row_offsets == 0,
            pid_m,
            num_assignments,
        )
    else:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return
        assignment_offsets = pid_m * BLOCK_SIZE_M + row_offsets
        assignment_ids = tl.load(sorted_token_ids_ptr + assignment_offsets)
    assignment_ids = assignment_ids.to(tl.int64)
    assignment_mask = assignment_ids < num_assignments

    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if expert_id < 0:
        return

    n_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    input_rows = assignment_ids // INPUT_TOP_K
    a_ptrs = (
        a_ptr
        + input_rows[:, None] * stride_am
        + k_offsets[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + expert_id * stride_be
        + n_offsets[None, :] * stride_bn
        + k_offsets[:, None] * stride_bk
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k_start * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=assignment_mask[:, None] & (k_offsets[None, :] < remaining_k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(k_offsets[:, None] < remaining_k)
            & (n_offsets[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTING_WEIGHT:
        routing_weights = tl.load(
            routing_weights_ptr + assignment_ids,
            mask=assignment_mask,
            other=0.0,
        ).to(tl.float32)
        accumulator *= routing_weights[:, None]

    c_ptrs = (
        c_ptr
        + assignment_ids[:, None] * stride_cm
        + n_offsets[None, :] * stride_cn
    )
    tl.store(
        c_ptrs,
        accumulator,
        mask=assignment_mask[:, None] & (n_offsets[None, :] < N),
    )


def _routed_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    alignment: MoeAlignment,
    *,
    input_top_k: int,
    multiply_routing_weight: bool,
    launch_config: dict[str, int],
) -> None:
    num_assignments = int(topk_weights.numel())
    if alignment.naive:
        em = num_assignments * alignment.block_size
        sorted_token_ids = topk_weights
    else:
        if alignment.sorted_token_ids is None:
            raise RuntimeError("Aligned routed GEMM is missing sorted_token_ids.")
        em = int(alignment.sorted_token_ids.numel())
        sorted_token_ids = alignment.sorted_token_ids

    block_m = launch_config["BLOCK_SIZE_M"]
    block_n = launch_config["BLOCK_SIZE_N"]
    grid = (
        triton.cdiv(em, block_m) * triton.cdiv(int(weights.shape[1]), block_n),
    )
    _routed_gemm_kernel[grid](
        inputs,
        weights,
        output,
        topk_weights,
        sorted_token_ids,
        alignment.expert_ids,
        alignment.num_tokens_post_padded,
        N=int(weights.shape[1]),
        K=int(weights.shape[2]),
        EM=em,
        num_assignments=num_assignments,
        stride_am=inputs.stride(0),
        stride_ak=inputs.stride(1),
        stride_be=weights.stride(0),
        stride_bn=weights.stride(1),
        stride_bk=weights.stride(2),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        INPUT_TOP_K=int(input_top_k),
        MUL_ROUTING_WEIGHT=bool(multiply_routing_weight),
        NAIVE_ASSIGNMENT=alignment.naive,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=launch_config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=launch_config["GROUP_SIZE_M"],
        num_warps=launch_config["num_warps"],
        num_stages=launch_config["num_stages"],
    )


@triton.jit
def _moe_sum_kernel(
    inputs_ptr,
    output_ptr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    stride_im,
    stride_ik,
    stride_in,
    stride_om,
    stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    token_offsets = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    hidden_offsets = tl.program_id(1) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask = (token_offsets[:, None] < num_tokens) & (
        hidden_offsets[None, :] < hidden_size
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for topk_slot in tl.static_range(top_k):
        values = tl.load(
            inputs_ptr
            + token_offsets[:, None] * stride_im
            + topk_slot * stride_ik
            + hidden_offsets[None, :] * stride_in,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        accumulator += values
    tl.store(
        output_ptr
        + token_offsets[:, None] * stride_om
        + hidden_offsets[None, :] * stride_on,
        accumulator,
        mask=mask,
    )


def _moe_sum(inputs: torch.Tensor) -> torch.Tensor:
    num_tokens, top_k, hidden_size = (int(dim) for dim in inputs.shape)
    block_m = 1 if num_tokens <= 4 else (4 if num_tokens <= 128 else 8)
    block_n = 256
    output = torch.empty(
        (num_tokens, hidden_size),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    grid = (
        triton.cdiv(num_tokens, block_m),
        triton.cdiv(hidden_size, block_n),
    )
    _moe_sum_kernel[grid](
        inputs,
        output,
        num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        stride_im=inputs.stride(0),
        stride_ik=inputs.stride(1),
        stride_in=inputs.stride(2),
        stride_om=output.stride(0),
        stride_on=output.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        num_warps=4 if block_m <= 4 else 8,
        num_stages=2,
    )
    return output


def _validate_fused_moe_inputs(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_expert_start: int,
) -> None:
    tensors = {
        "hidden_states": hidden_states,
        "w13_weight": w13_weight,
        "w2_weight": w2_weight,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
    }
    for name, tensor in tensors.items():
        if not tensor.is_cuda:
            raise ValueError(f"Triton MoE requires CUDA {name}.")
        if tensor.device != hidden_states.device:
            raise ValueError(
                f"All Triton MoE tensors must share one device; {name} is on "
                f"{tensor.device}, hidden_states is on {hidden_states.device}."
            )
    if hidden_states.ndim != 2 or hidden_states.shape[0] <= 0:
        raise ValueError(
            "hidden_states must have non-empty shape [tokens, hidden], got "
            f"{tuple(hidden_states.shape)}."
        )
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise ValueError(
            "Expert weights must have shapes [local_experts, 2*intermediate, hidden] "
            "and [local_experts, hidden, intermediate]."
        )
    if topk_ids.ndim != 2 or topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "topk_ids and topk_weights must have the same [tokens, top_k] shape, "
            f"got ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}."
        )
    if int(topk_ids.shape[0]) != int(hidden_states.shape[0]):
        raise ValueError(
            "Router token count does not match hidden_states: "
            f"{topk_ids.shape[0]} != {hidden_states.shape[0]}."
        )
    if int(topk_ids.shape[1]) <= 0 or int(topk_ids.shape[1]) > int(num_experts):
        raise ValueError(
            f"top_k must be in [1, {num_experts}], got {topk_ids.shape[1]}."
        )
    if hidden_states.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "Triton MoE supports BF16 and FP16 activations, got "
            f"dtype={hidden_states.dtype}."
        )
    if w13_weight.dtype != hidden_states.dtype or w2_weight.dtype != hidden_states.dtype:
        raise TypeError(
            "Triton MoE activations and expert weights must share BF16/FP16 dtype, got "
            f"hidden={hidden_states.dtype}, w13={w13_weight.dtype}, w2={w2_weight.dtype}."
        )
    if topk_weights.dtype not in (*_SUPPORTED_DTYPES, torch.float32):
        raise TypeError(
            f"topk_weights must use BF16, FP16, or FP32, got {topk_weights.dtype}."
        )
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"topk_ids must use int32 or int64, got {topk_ids.dtype}.")
    for name, tensor in tensors.items():
        if not tensor.is_contiguous():
            raise ValueError(f"Triton MoE requires contiguous {name}.")

    num_local_experts = int(w13_weight.shape[0])
    hidden_size = int(hidden_states.shape[1])
    if num_local_experts <= 0:
        raise ValueError("Triton MoE requires at least one local expert.")
    if int(w2_weight.shape[0]) != num_local_experts:
        raise ValueError("w13_weight and w2_weight local expert counts differ.")
    if int(w13_weight.shape[1]) % 2 != 0:
        raise ValueError("w13_weight output size must be even for SiLU-and-mul.")
    intermediate_size = int(w13_weight.shape[1]) // 2
    if int(w13_weight.shape[2]) != hidden_size:
        raise ValueError("w13_weight input size does not match hidden_states.")
    if tuple(w2_weight.shape[1:]) != (hidden_size, intermediate_size):
        raise ValueError(
            "w2_weight shape does not match hidden/intermediate sizes: expected "
            f"({num_local_experts}, {hidden_size}, {intermediate_size}), got "
            f"{tuple(w2_weight.shape)}."
        )
    local_expert_end = int(local_expert_start) + num_local_experts
    if not 0 <= int(local_expert_start) < local_expert_end <= int(num_experts):
        raise ValueError(
            f"Invalid local expert range [{local_expert_start}, {local_expert_end}) "
            f"for num_experts={num_experts}."
        )


def fused_moe(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
) -> torch.Tensor:
    """Run unquantized routed experts with a generic Triton MoE pipeline."""

    num_experts = int(num_experts)
    local_expert_start = int(local_expert_start)
    _validate_fused_moe_inputs(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts,
        local_expert_start,
    )

    num_tokens = int(hidden_states.shape[0])
    top_k = int(topk_ids.shape[1])
    num_assignments = num_tokens * top_k
    intermediate_size = int(w13_weight.shape[1]) // 2
    hidden_size = int(hidden_states.shape[1])
    local_expert_end = local_expert_start + int(w13_weight.shape[0])

    w13_config = _gemm_launch_config(num_assignments, 2 * intermediate_size)
    alignment = _prepare_expert_assignment(
        topk_ids,
        block_size=w13_config["BLOCK_SIZE_M"],
        num_experts=num_experts,
        local_expert_start=local_expert_start,
        local_expert_end=local_expert_end,
    )

    w13_output = torch.zeros(
        (num_assignments, 2 * intermediate_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        hidden_states,
        w13_weight,
        w13_output,
        topk_weights,
        alignment,
        input_top_k=top_k,
        multiply_routing_weight=False,
        launch_config=w13_config,
    )
    activated = silu_and_mul_fwd(w13_output)

    w2_config = dict(_gemm_launch_config(num_assignments, hidden_size))
    w2_config["BLOCK_SIZE_M"] = alignment.block_size
    w2_output = torch.zeros(
        (num_assignments, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    _routed_gemm(
        activated,
        w2_weight,
        w2_output,
        topk_weights,
        alignment,
        input_top_k=1,
        multiply_routing_weight=True,
        launch_config=w2_config,
    )
    return _moe_sum(w2_output.view(num_tokens, top_k, hidden_size))
