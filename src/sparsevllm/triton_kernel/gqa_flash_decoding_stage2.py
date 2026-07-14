import os

import torch
import triton
import triton.language as tl


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
    BLOCK_COUNT: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_len = tl.load(B_Seqlen + cur_batch)
    num_blocks = tl.maximum((seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ, 0)

    lse_base = cur_batch * stride_mid_o_eb + cur_head * stride_mid_o_eh
    offs_d = tl.arange(0, HEAD_DIM)
    mid_base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh
    if BLOCK_COUNT <= 4:
        max_lse = -float("inf")
        weight_sum = 0.0
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        for block_idx in range(0, num_blocks, 1):
            value = tl.load(
                Mid_O
                + mid_base
                + block_idx * stride_mid_os
                + offs_d * stride_mid_od
            )
            lse = tl.load(Mid_O_LogExpSum + lse_base + block_idx * stride_mid_o_es)
            new_max = tl.maximum(max_lse, lse)
            old_scale = tl.exp(max_lse - new_max)
            new_weight = tl.exp(lse - new_max)
            acc = acc * old_scale + new_weight * value
            weight_sum = weight_sum * old_scale + new_weight
            max_lse = new_max
        safe_weight_sum = tl.where(num_blocks > 0, weight_sum, 1.0)
        output = tl.where(num_blocks > 0, acc / safe_weight_sum, 0.0)
    else:
        block_offsets = tl.arange(0, BLOCK_COUNT)
        block_mask = block_offsets < num_blocks
        block_lse = tl.load(
            Mid_O_LogExpSum + lse_base + block_offsets * stride_mid_o_es,
            mask=block_mask,
            other=-float("inf"),
        )
        global_max = tl.max(block_lse, axis=0)
        safe_max = tl.where(num_blocks > 0, global_max, 0.0)
        block_weights = tl.where(block_mask, tl.exp(block_lse - safe_max), 0.0)
        weight_sum = tl.sum(block_weights, axis=0)

        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
        chunk_offsets = tl.arange(0, BLOCK_M)
        for block_start in range(0, num_blocks, BLOCK_M):
            block_indices = block_start + chunk_offsets
            chunk_mask = block_indices < num_blocks
            values = tl.load(
                Mid_O
                + mid_base
                + block_indices[:, None] * stride_mid_os
                + offs_d[None, :] * stride_mid_od,
                mask=chunk_mask[:, None],
                other=0.0,
            )
            chunk_lse = tl.load(
                Mid_O_LogExpSum + lse_base + block_indices * stride_mid_o_es,
                mask=chunk_mask,
                other=-float("inf"),
            )
            chunk_weights = tl.exp(chunk_lse - safe_max)
            acc += tl.sum(chunk_weights[:, None] * values, axis=0)
        output = tl.where(num_blocks > 0, acc / weight_sum, 0.0)
    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d * stride_od,
        output,
    )


def _validate_inputs_uncached(mid_out, mid_out_logexpsum, b_seqlen, output, block_seq):
    tensors = (mid_out, mid_out_logexpsum, b_seqlen, output)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("GQA flash decode stage2 expects all tensors on CUDA.")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"GQA flash decode stage2 tensors must share one device, got {devices}.")
    if mid_out.ndim != 4 or mid_out_logexpsum.ndim != 3 or output.ndim != 3:
        raise ValueError(
            "mid_out/mid_out_logexpsum/output must have ranks 4/3/3, "
            f"got {mid_out.ndim}/{mid_out_logexpsum.ndim}/{output.ndim}."
        )
    batch, num_heads, num_blocks, head_dim = map(int, mid_out.shape)
    supported_head_dims = (16, 32, 64, 128, 256)
    if head_dim not in supported_head_dims:
        raise ValueError(f"head_dim must be one of {supported_head_dims}, got {head_dim}.")
    if tuple(mid_out_logexpsum.shape) != (batch, num_heads, num_blocks):
        raise ValueError(
            f"mid_out_logexpsum must have shape {(batch, num_heads, num_blocks)}, "
            f"got {tuple(mid_out_logexpsum.shape)}."
        )
    if tuple(output.shape) != (batch, num_heads, head_dim):
        raise ValueError(
            f"output must have shape {(batch, num_heads, head_dim)}, got {tuple(output.shape)}."
        )
    if b_seqlen.ndim != 1 or len(b_seqlen) != batch:
        raise ValueError(f"b_seqlen must have shape {(batch,)}, got {tuple(b_seqlen.shape)}.")
    if b_seqlen.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"b_seqlen must use int32 or int64, got {b_seqlen.dtype}.")
    if mid_out.dtype != torch.float32 or mid_out_logexpsum.dtype != torch.float32:
        raise ValueError(
            f"decode workspaces must be float32, got {mid_out.dtype}/{mid_out_logexpsum.dtype}."
        )
    if not output.is_floating_point():
        raise ValueError(f"output must use a floating dtype, got {output.dtype}.")
    if int(block_seq) <= 0:
        raise ValueError(f"block_seq must be positive, got {block_seq}.")
    return batch, num_heads, num_blocks, head_dim


_VALIDATED_INPUT_SPECS = {}


def _tensor_spec(tensor):
    return tensor.device, tensor.dtype, tensor.shape, tensor.stride()


def _validate_inputs(mid_out, mid_out_logexpsum, b_seqlen, output, block_seq):
    debug_bounds = os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1"
    if debug_bounds:
        return _validate_inputs_uncached(
            mid_out, mid_out_logexpsum, b_seqlen, output, block_seq
        )
    spec = (
        _tensor_spec(mid_out),
        _tensor_spec(mid_out_logexpsum),
        _tensor_spec(b_seqlen),
        _tensor_spec(output),
        int(block_seq),
    )
    validated = _VALIDATED_INPUT_SPECS.get(spec)
    if validated is None:
        validated = _validate_inputs_uncached(
            mid_out, mid_out_logexpsum, b_seqlen, output, block_seq
        )
        if len(_VALIDATED_INPUT_SPECS) >= 128:
            _VALIDATED_INPUT_SPECS.clear()
        _VALIDATED_INPUT_SPECS[spec] = validated
    return validated


@torch.no_grad()
def flash_decode_stage2(mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq):
    batch, num_heads, num_blocks, head_dim = _validate_inputs(
        mid_out, mid_out_logexpsum, B_Seqlen, O, block_seq
    )
    if batch == 0:
        return
    if num_blocks == 0:
        O.zero_()
        return

    if (
        os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1"
        and not torch.cuda.is_current_stream_capturing()
    ):
        min_seq_len = int(B_Seqlen.min().item()) if B_Seqlen.numel() > 0 else 0
        max_seq_len = int(B_Seqlen.max().item()) if B_Seqlen.numel() > 0 else 0
        needed_blocks = triton.cdiv(max_seq_len, int(block_seq))
        if min_seq_len < 0 or needed_blocks > num_blocks:
            raise RuntimeError(
                "GQA flash decode stage2 bounds check failed: "
                f"seq_len_range=[{min_seq_len}, {max_seq_len}] block_seq={block_seq} "
                f"needed_blocks={needed_blocks} workspace_blocks={num_blocks}."
            )

    block_count = triton.next_power_of_2(num_blocks)
    _fwd_kernel_flash_decode_stage2[(batch, num_heads)](
        B_Seqlen,
        mid_out,
        mid_out_logexpsum,
        O,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logexpsum.stride(0),
        mid_out_logexpsum.stride(1),
        mid_out_logexpsum.stride(2),
        O.stride(0),
        O.stride(1),
        O.stride(2),
        BLOCK_SEQ=int(block_seq),
        HEAD_DIM=head_dim,
        BLOCK_COUNT=block_count,
        BLOCK_M=min(16, block_count),
        num_warps=4 if head_dim <= 128 or block_count <= 16 else 8,
        num_stages=2,
    )
