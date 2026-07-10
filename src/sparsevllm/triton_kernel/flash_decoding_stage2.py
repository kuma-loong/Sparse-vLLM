import os

import torch
import triton
import triton.language as tl


_MIN_HEAD_DIM = 16
_MAX_HEAD_DIM = 256
_VARIANT_WARPS = {"unified_s2_w4": 4, "unified_s2_w8": 8}


@triton.jit
def _fwd_kernel_flash_decode_stage2(
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    O,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_obs,
    stride_oh,
    stride_od,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    dim_mask = offs_d < HEAD_DIM
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    block_n_size = tl.where(cur_batch_seq_len > 0, cur_batch_seq_len + BLOCK_SEQ - 1, 0) // BLOCK_SEQ

    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    offs_v = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + offs_d * stride_mid_od
    )
    offs_logic = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    for block_seq_n in range(0, block_n_size, 1):
        block_o = tl.load(
            Mid_O + offs_v + block_seq_n * stride_mid_os,
            mask=dim_mask,
            other=0.0,
        )
        block_lse = tl.load(Mid_O_LogExpSum + offs_logic + block_seq_n * stride_mid_o_es)
        new_max_logic = tl.maximum(block_lse, max_logic)
        old_scale = tl.exp(max_logic - new_max_logic)
        acc *= old_scale
        exp_logic = tl.exp(block_lse - new_max_logic)
        acc += exp_logic * block_o
        sum_exp = sum_exp * old_scale + exp_logic
        max_logic = new_max_logic

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od,
        acc / sum_exp,
        mask=(block_n_size > 0) & dim_mask,
    )


def _validate_inputs(mid_out, mid_out_logexpsum, b_seqlen, output, block_seq):
    tensors = (mid_out, mid_out_logexpsum, b_seqlen, output)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("flash_decode_stage2 tensors must all be CUDA tensors.")
    if any(t.device != mid_out.device for t in tensors):
        raise ValueError(f"stage 2 tensors must share one CUDA device, got {[str(t.device) for t in tensors]}.")
    if mid_out.dim() != 4 or mid_out_logexpsum.dim() != 3 or output.dim() != 3:
        raise ValueError(
            f"stage 2 expects mid_out/mid_lse/output ranks 4/3/3, got "
            f"{mid_out.shape}/{mid_out_logexpsum.shape}/{output.shape}."
        )
    batch, heads, blocks, head_dim = map(int, mid_out.shape)
    if not (_MIN_HEAD_DIM <= head_dim <= _MAX_HEAD_DIM):
        raise ValueError(f"stage 2 head_dim must be in [{_MIN_HEAD_DIM}, {_MAX_HEAD_DIM}], got {head_dim}.")
    if mid_out.dtype != torch.float32 or mid_out_logexpsum.dtype != torch.float32:
        raise ValueError(
            f"stage 2 workspace must be FP32, got {mid_out.dtype}/{mid_out_logexpsum.dtype}."
        )
    if tuple(mid_out_logexpsum.shape[:2]) != (batch, heads) or mid_out_logexpsum.shape[2] < blocks:
        raise ValueError(
            f"mid_out_logexpsum shape {tuple(mid_out_logexpsum.shape)} is incompatible with "
            f"mid_out {tuple(mid_out.shape)}."
        )
    if tuple(output.shape) != (batch, heads, head_dim):
        raise ValueError(f"output must have shape {(batch, heads, head_dim)}, got {tuple(output.shape)}.")
    if b_seqlen.dim() != 1 or b_seqlen.numel() != batch:
        raise ValueError(f"B_Seqlen must be rank-1 with {batch} rows, got {tuple(b_seqlen.shape)}.")
    if b_seqlen.dtype not in {torch.int32, torch.int64}:
        raise ValueError(f"B_Seqlen must use int32 or int64, got {b_seqlen.dtype}.")
    block_seq = int(block_seq)
    if block_seq <= 0 or block_seq % 16 != 0:
        raise ValueError(f"block_seq must be positive and divisible by 16, got {block_seq}.")
    if os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1" and not torch.cuda.is_current_stream_capturing():
        min_seq_len = int(b_seqlen.min().item()) if b_seqlen.numel() else 0
        max_seq_len = int(b_seqlen.max().item()) if b_seqlen.numel() else 0
        needed_blocks = (max_seq_len + block_seq - 1) // block_seq
        if min_seq_len <= 0:
            raise ValueError(
                f"flash_decode_stage2 requires positive context lengths; got minimum {min_seq_len}."
            )
        if blocks < needed_blocks or mid_out_logexpsum.shape[2] < needed_blocks:
            raise RuntimeError(
                "flash_decode_stage2 bounds check failed: "
                f"max_seq_len={max_seq_len} block_seq={block_seq} needed_blocks={needed_blocks} "
                f"mid_out_blocks={blocks} mid_lse_blocks={mid_out_logexpsum.shape[2]}."
            )


@torch.no_grad()
def flash_decode_stage2_variant(
    mid_out,
    mid_out_logexpsum,
    b_seqlen,
    output,
    block_seq,
    *,
    variant_id,
):
    """Research A/B entry point; production code uses flash_decode_stage2."""
    _validate_inputs(mid_out, mid_out_logexpsum, b_seqlen, output, block_seq)
    try:
        num_warps = _VARIANT_WARPS[variant_id]
    except KeyError as exc:
        raise ValueError(f"Unknown stage 2 variant_id={variant_id!r}; expected one of {sorted(_VARIANT_WARPS)}.") from exc
    grid = (mid_out.shape[0], mid_out.shape[1])
    _fwd_kernel_flash_decode_stage2[grid](
        b_seqlen,
        mid_out,
        mid_out_logexpsum,
        output,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logexpsum.stride(0),
        mid_out_logexpsum.stride(1),
        mid_out_logexpsum.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BLOCK_SEQ=block_seq,
        HEAD_DIM=mid_out.shape[-1],
        BLOCK_DMODEL=triton.next_power_of_2(mid_out.shape[-1]),
        num_warps=num_warps,
        num_stages=2,
    )


@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, b_seqlen, output, block_seq):
    padded_head_dim = triton.next_power_of_2(mid_out.shape[-1])
    variant_id = "unified_s2_w8" if padded_head_dim >= 256 else "unified_s2_w4"
    flash_decode_stage2_variant(
        mid_out,
        mid_out_logexpsum,
        b_seqlen,
        output,
        block_seq,
        variant_id=variant_id,
    )
