import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_flash_decode_stage1(
    Q,
    K,
    V,
    sm_scale,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    Mid_O,
    Mid_O_LogExpSum,
    Attn_Score,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_mid_od,
    stride_mid_o_eb,
    stride_mid_o_eh,
    stride_mid_o_es,
    stride_asbs,
    stride_ash,
    stride_asl,
    GQA_GROUP_SIZE: tl.constexpr,
    Q_TILES_PER_KV_HEAD: tl.constexpr,
    STORE_SCORE_3D: tl.constexpr,
    STORE_SCORE_2D: tl.constexpr,
    Q_HEAD_TILE: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_q_tile = tl.program_id(1)
    cur_kv_head = cur_q_tile // Q_TILES_PER_KV_HEAD
    tile_in_kv_head = cur_q_tile - cur_kv_head * Q_TILES_PER_KV_HEAD
    seq_start_block = tl.program_id(2)

    q_head_offsets = tl.arange(0, Q_HEAD_TILE)
    group_offsets = tile_in_kv_head * Q_HEAD_TILE + q_head_offsets
    q_heads = cur_kv_head * GQA_GROUP_SIZE + group_offsets
    head_mask = group_offsets < GQA_GROUP_SIZE
    offs_d = tl.arange(0, HEAD_DIM)

    seq_len = tl.load(B_Seqlen + cur_batch)
    req_idx = tl.load(B_req_idx + cur_batch)
    seq_start = seq_start_block * BLOCK_SEQ
    seq_end = tl.minimum(seq_len, seq_start + BLOCK_SEQ)
    tokens_in_block = tl.maximum(seq_end - seq_start, 0)
    num_token_tiles = (tokens_in_block + BLOCK_N - 1) // BLOCK_N

    q_offsets = (
        cur_batch * stride_qbs
        + q_heads[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(
        Q + q_offsets,
        mask=head_mask[:, None],
        other=0.0,
    )

    base_token_offsets = seq_start + tl.arange(0, BLOCK_N)
    sum_exp = tl.zeros([Q_HEAD_TILE], dtype=tl.float32)
    max_logic = tl.full([Q_HEAD_TILE], -float("inf"), dtype=tl.float32)
    acc = tl.zeros([Q_HEAD_TILE, HEAD_DIM], dtype=tl.float32)

    for tile_idx in range(0, num_token_tiles, 1):
        token_offsets = tile_idx * BLOCK_N + base_token_offsets
        token_mask = token_offsets < seq_end
        k_loc = tl.load(
            Req_to_tokens
            + req_idx * stride_req_to_tokens_b
            + token_offsets * stride_req_to_tokens_s,
            mask=token_mask,
            other=0,
        )

        k_offsets = (
            k_loc[None, :] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(
            K + k_offsets,
            mask=token_mask[None, :],
            other=0.0,
        )
        att_value = tl.dot(q, k)
        att_value = tl.where(token_mask[None, :], att_value, -float("inf"))

        if STORE_SCORE_3D:
            score_offsets = (
                cur_batch * stride_asbs
                + q_heads[:, None] * stride_ash
                + token_offsets[None, :] * stride_asl
            )
            tl.store(
                Attn_Score + score_offsets,
                att_value,
                mask=head_mask[:, None] & token_mask[None, :],
            )
        if STORE_SCORE_2D:
            score_by_token = tl.max(
                tl.where(head_mask[:, None], att_value, -float("inf")),
                axis=0,
            )
            tl.atomic_max(
                Attn_Score
                + cur_batch * stride_asbs
                + token_offsets * stride_asl,
                score_by_token,
                mask=token_mask,
            )

        att_value *= sm_scale
        v_offsets = (
            k_loc[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(
            V + v_offsets,
            mask=token_mask[:, None],
            other=0.0,
        )

        tile_max = tl.max(att_value, axis=1)
        new_max = tl.maximum(max_logic, tile_max)
        old_scale = tl.exp(max_logic - new_max)
        probabilities = tl.exp(att_value - new_max[:, None])
        acc *= old_scale[:, None]
        acc += tl.dot(probabilities.to(v.dtype), v)
        sum_exp = sum_exp * old_scale + tl.sum(probabilities, axis=1)
        max_logic = new_max

    has_tokens = tokens_in_block > 0
    safe_sum_exp = tl.where(has_tokens, sum_exp, 1.0)
    mid_offsets = (
        cur_batch * stride_mid_ob
        + q_heads[:, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, :] * stride_mid_od
    )
    tl.store(
        Mid_O + mid_offsets,
        tl.where(has_tokens, acc / safe_sum_exp[:, None], 0.0),
        mask=head_mask[:, None],
    )
    lse_offsets = (
        cur_batch * stride_mid_o_eb
        + q_heads * stride_mid_o_eh
        + seq_start_block * stride_mid_o_es
    )
    tl.store(
        Mid_O_LogExpSum + lse_offsets,
        tl.where(has_tokens, max_logic + tl.log(safe_sum_exp), -float("inf")),
        mask=head_mask,
    )


def _validate_inputs_uncached(
    q,
    k,
    v,
    req_to_tokens,
    b_req_idx,
    b_seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    attn_score=None,
):
    tensors = (q, k, v, req_to_tokens, b_req_idx, b_seqlen, mid_out, mid_out_logsumexp)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("GQA flash decode expects all input and workspace tensors on CUDA.")
    devices = {tensor.device for tensor in tensors}
    if len(devices) != 1:
        raise ValueError(f"GQA flash decode tensors must share one CUDA device, got {devices}.")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            f"q/k/v must be rank-3, got shapes {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}."
        )
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(f"q/k/v dtypes must match, got {q.dtype}/{k.dtype}/{v.dtype}.")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"q/k/v must use float16, bfloat16, or float32, got {q.dtype}.")
    head_dim = int(q.shape[2])
    supported_head_dims = (16, 32, 64, 128, 256)
    if head_dim not in supported_head_dims or int(k.shape[2]) != head_dim or int(v.shape[2]) != head_dim:
        raise ValueError(
            f"q/k/v head dimensions must match and be one of {supported_head_dims}, "
            f"got {q.shape[2]}/{k.shape[2]}/{v.shape[2]}."
        )
    num_q_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    if num_kv_heads <= 0 or int(v.shape[1]) != num_kv_heads:
        raise ValueError(f"k/v head counts must be positive and equal, got {k.shape[1]}/{v.shape[1]}.")
    if num_q_heads <= num_kv_heads or num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "GQA flash decode requires q_heads > kv_heads and exact grouping, "
            f"got q_heads={num_q_heads} kv_heads={num_kv_heads}."
        )
    batch = int(q.shape[0])
    if b_req_idx.ndim != 1 or b_seqlen.ndim != 1 or len(b_req_idx) != batch or len(b_seqlen) != batch:
        raise ValueError(
            "b_req_idx and b_seqlen must be rank-1 tensors with q batch length, "
            f"got q_batch={batch} shapes={tuple(b_req_idx.shape)}/{tuple(b_seqlen.shape)}."
        )
    if req_to_tokens.ndim != 2:
        raise ValueError(f"req_to_tokens must be rank-2, got shape={tuple(req_to_tokens.shape)}.")
    integer_dtypes = (torch.int32, torch.int64)
    if (
        req_to_tokens.dtype not in integer_dtypes
        or b_req_idx.dtype not in integer_dtypes
        or b_seqlen.dtype not in integer_dtypes
    ):
        raise ValueError(
            "req_to_tokens, b_req_idx, and b_seqlen must use int32 or int64, "
            f"got {req_to_tokens.dtype}/{b_req_idx.dtype}/{b_seqlen.dtype}."
        )
    max_len_in_batch = int(max_len_in_batch)
    block_seq = int(block_seq)
    if max_len_in_batch < 0:
        raise ValueError(f"max_len_in_batch must be non-negative, got {max_len_in_batch}.")
    if block_seq <= 0:
        raise ValueError(f"block_seq must be positive, got {block_seq}.")
    if int(req_to_tokens.shape[1]) < max_len_in_batch:
        raise ValueError(
            f"req_to_tokens width {req_to_tokens.shape[1]} is smaller than max_len_in_batch {max_len_in_batch}."
        )
    num_blocks = triton.cdiv(max_len_in_batch, block_seq)
    expected_mid_shape = (batch, num_q_heads, num_blocks)
    if (
        mid_out.ndim != 4
        or tuple(mid_out.shape[:2]) != expected_mid_shape[:2]
        or int(mid_out.shape[2]) < num_blocks
        or int(mid_out.shape[3]) < head_dim
    ):
        raise ValueError(
            f"mid_out must cover [batch, q_heads, blocks, head_dim]={expected_mid_shape + (head_dim,)}, "
            f"got {tuple(mid_out.shape)}."
        )
    if (
        mid_out_logsumexp.ndim != 3
        or tuple(mid_out_logsumexp.shape[:2]) != expected_mid_shape[:2]
        or int(mid_out_logsumexp.shape[2]) < num_blocks
    ):
        raise ValueError(
            f"mid_out_logsumexp must cover {expected_mid_shape}, got {tuple(mid_out_logsumexp.shape)}."
        )
    if mid_out.dtype != torch.float32 or mid_out_logsumexp.dtype != torch.float32:
        raise ValueError(
            f"decode workspaces must be float32, got {mid_out.dtype}/{mid_out_logsumexp.dtype}."
        )
    if attn_score is not None:
        if not attn_score.is_cuda or attn_score.ndim not in (2, 3):
            raise ValueError(
                f"attn_score must be a rank-2 or rank-3 CUDA tensor, got shape={tuple(attn_score.shape)}."
            )
        if int(attn_score.shape[0]) != batch or int(attn_score.shape[-1]) < max_len_in_batch:
            raise ValueError(
                f"attn_score must cover batch={batch} and length={max_len_in_batch}, got {tuple(attn_score.shape)}."
            )
        if attn_score.ndim == 3 and int(attn_score.shape[1]) != num_q_heads:
            raise ValueError(
                f"rank-3 attn_score must have {num_q_heads} heads, got {attn_score.shape[1]}."
            )
        if attn_score.device != q.device:
            raise ValueError(
                f"attn_score must be on {q.device}, got {attn_score.device}."
            )
    if (
        os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1"
        and not torch.cuda.is_current_stream_capturing()
        and batch > 0
    ):
        min_len = int(b_seqlen.min().item())
        max_len = int(b_seqlen.max().item())
        min_req = int(b_req_idx.min().item())
        max_req = int(b_req_idx.max().item())
        if min_len < 0 or max_len > max_len_in_batch:
            raise RuntimeError(
                "GQA flash decode sequence bounds check failed: "
                f"seq_len_range=[{min_len}, {max_len}] max_len_in_batch={max_len_in_batch}."
            )
        if min_req < 0 or max_req >= int(req_to_tokens.shape[0]):
            raise RuntimeError(
                "GQA flash decode request bounds check failed: "
                f"req_range=[{min_req}, {max_req}] table_rows={req_to_tokens.shape[0]}."
            )
        if max_len > 0:
            rows = b_req_idx.to(torch.long)
            visible_slots = req_to_tokens.index_select(0, rows)[:, :max_len]
            positions = torch.arange(max_len, device=q.device)[None, :]
            valid = positions < b_seqlen[:, None]
            slot_capacity = min(int(k.shape[0]), int(v.shape[0]))
            invalid = ((visible_slots < 0) | (visible_slots >= slot_capacity)) & valid
            if bool(invalid.any().item()):
                location = invalid.nonzero(as_tuple=False)[0]
                invalid_batch = int(location[0].item())
                invalid_position = int(location[1].item())
                invalid_slot = int(visible_slots[invalid_batch, invalid_position].item())
                raise RuntimeError(
                    "GQA flash decode slot bounds check failed: "
                    f"batch={invalid_batch} position={invalid_position} slot={invalid_slot} "
                    f"slot_capacity={slot_capacity}."
                )
    return max_len_in_batch, block_seq, head_dim, num_q_heads, num_kv_heads, num_blocks


_VALIDATED_INPUT_SPECS = {}


def _tensor_spec(tensor):
    return tensor.device, tensor.dtype, tensor.shape, tensor.stride()


def _validate_inputs(
    q,
    k,
    v,
    req_to_tokens,
    b_req_idx,
    b_seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    attn_score=None,
):
    debug_bounds = os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") == "1"
    if debug_bounds:
        return _validate_inputs_uncached(
            q,
            k,
            v,
            req_to_tokens,
            b_req_idx,
            b_seqlen,
            max_len_in_batch,
            mid_out,
            mid_out_logsumexp,
            block_seq,
            attn_score,
        )
    spec = (
        _tensor_spec(q),
        _tensor_spec(k),
        _tensor_spec(v),
        _tensor_spec(req_to_tokens),
        _tensor_spec(b_req_idx),
        _tensor_spec(b_seqlen),
        _tensor_spec(mid_out),
        _tensor_spec(mid_out_logsumexp),
        _tensor_spec(attn_score) if attn_score is not None else None,
        int(max_len_in_batch),
        int(block_seq),
    )
    validated = _VALIDATED_INPUT_SPECS.get(spec)
    if validated is None:
        validated = _validate_inputs_uncached(
            q,
            k,
            v,
            req_to_tokens,
            b_req_idx,
            b_seqlen,
            max_len_in_batch,
            mid_out,
            mid_out_logsumexp,
            block_seq,
            attn_score,
        )
        if len(_VALIDATED_INPUT_SPECS) >= 128:
            _VALIDATED_INPUT_SPECS.clear()
        _VALIDATED_INPUT_SPECS[spec] = validated
    return validated


def _kernel_config(head_dim: int, max_len_in_batch: int, total_programs: int):
    if head_dim >= 96 and total_programs <= 256:
        return 64, 2, 3
    if head_dim >= 96 and max_len_in_batch > 4096:
        return (64, 2, 2) if head_dim <= 128 else (16, 2, 2)
    return 16, 2, 2


def _launch_stage1(
    q,
    k,
    v,
    req_to_tokens,
    b_req_idx,
    b_seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
    attn_score=None,
):
    (
        max_len_in_batch,
        block_seq,
        head_dim,
        num_q_heads,
        num_kv_heads,
        num_blocks,
    ) = _validate_inputs(
        q,
        k,
        v,
        req_to_tokens,
        b_req_idx,
        b_seqlen,
        max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        block_seq,
        attn_score,
    )
    if q.shape[0] == 0 or num_blocks == 0:
        return

    group_size = num_q_heads // num_kv_heads
    q_head_tile = min(32, max(16, triton.next_power_of_2(group_size)))
    q_tiles_per_kv_head = triton.cdiv(group_size, q_head_tile)
    total_programs = q.shape[0] * num_kv_heads * q_tiles_per_kv_head * num_blocks
    block_n, num_warps, num_stages = _kernel_config(
        head_dim, max_len_in_batch, total_programs
    )
    score_3d = attn_score is not None and attn_score.ndim == 3
    score_2d = attn_score is not None and attn_score.ndim == 2
    score = mid_out if attn_score is None else attn_score
    stride_asbs = score.stride(0) if attn_score is not None else 0
    stride_ash = score.stride(1) if score_3d else 0
    stride_asl = score.stride(2) if score_3d else (score.stride(1) if score_2d else 0)

    grid = (q.shape[0], num_kv_heads * q_tiles_per_kv_head, num_blocks)
    _fwd_kernel_flash_decode_stage1[grid](
        q,
        k,
        v,
        head_dim**-0.5,
        req_to_tokens,
        b_req_idx,
        b_seqlen,
        mid_out,
        mid_out_logsumexp,
        score,
        req_to_tokens.stride(0),
        req_to_tokens.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_out.stride(3),
        mid_out_logsumexp.stride(0),
        mid_out_logsumexp.stride(1),
        mid_out_logsumexp.stride(2),
        stride_asbs,
        stride_ash,
        stride_asl,
        GQA_GROUP_SIZE=group_size,
        Q_TILES_PER_KV_HEAD=q_tiles_per_kv_head,
        STORE_SCORE_3D=score_3d,
        STORE_SCORE_2D=score_2d,
        Q_HEAD_TILE=q_head_tile,
        BLOCK_SEQ=block_seq,
        HEAD_DIM=head_dim,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )


@torch.no_grad()
def flash_decode_stage1(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    block_seq,
):
    _launch_stage1(
        q,
        k,
        v,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        block_seq,
    )


@torch.no_grad()
def flash_decode_stage1_with_score(
    q,
    k,
    v,
    Req_to_tokens,
    B_req_idx,
    B_Seqlen,
    max_len_in_batch,
    mid_out,
    mid_out_logsumexp,
    attn_score,
    block_seq,
):
    _launch_stage1(
        q,
        k,
        v,
        Req_to_tokens,
        B_req_idx,
        B_Seqlen,
        max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        block_seq,
        attn_score,
    )
