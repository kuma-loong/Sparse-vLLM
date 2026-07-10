import os

import torch
import triton
import triton.language as tl


_MIN_HEAD_DIM = 16
_MAX_HEAD_DIM = 256
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
_VARIANT_CONFIGS = {
    "grouped_s1_allow256_w2": ("grouped", 2, 16, 2),
    "grouped_s1_allow256_w4": ("grouped", 4, 16, 2),
    "grouped_s1_allow256_w8": ("grouped", 8, 16, 2),
    "grouped_s1_bn16_w2_s1": ("grouped", 2, 16, 1),
    "grouped_s1_bn16_w2_s2": ("grouped", 2, 16, 2),
    "grouped_s1_bn16_w2_s3": ("grouped", 2, 16, 3),
    "grouped_s1_bn32_w2_s2": ("grouped", 2, 32, 2),
    "grouped_s1_bn64_w2_s2": ("grouped", 2, 64, 2),
    "grouped_s1_bn128_w2_s2": ("grouped", 2, 128, 2),
    "per_q_s1_w4": ("per_q", 4, 16, 2),
    "per_q_s1_w8": ("per_q", 8, 16, 2),
}


def get_stage1_variant_config(variant_id):
    try:
        schedule, num_warps, block_n, num_stages = _VARIANT_CONFIGS[variant_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GQA stage 1 variant_id={variant_id!r}; expected one of {sorted(_VARIANT_CONFIGS)}."
        ) from exc
    return {
        "schedule": schedule,
        "num_warps": num_warps,
        "block_n": block_n,
        "num_stages": num_stages,
    }


def _requires_64bit_cache_offsets(*tensors):
    int32_max = torch.iinfo(torch.int32).max
    for tensor in tensors:
        max_offset = sum((size - 1) * abs(stride) for size, stride in zip(tensor.shape, tensor.stride()))
        if max_offset > int32_max:
            return True
    return False


@triton.jit
def _fwd_kernel_gqa_flash_decode_stage1_grouped(
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
    gqa_group_size,
    STORE_SCORE_3D: tl.constexpr,
    STORE_SCORE_2D: tl.constexpr,
    USE_64BIT_OFFSETS: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_kv_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)

    cur_q_head_range = cur_kv_head * gqa_group_size + tl.arange(0, Q_HEAD_NUM)
    head_mask = cur_q_head_range < (cur_kv_head + 1) * gqa_group_size
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)
    block_n_size = (
        tl.where(
            cur_batch_end_index > cur_batch_start_index,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
            0,
        )
        // BLOCK_N
    )

    off_q = (
        cur_batch * stride_qbs
        + cur_q_head_range[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    dim_mask = offs_d < HEAD_DIM
    q = tl.load(Q + off_q, mask=head_mask[:, None] & dim_mask[None, :], other=0.0)
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)

    sum_exp = tl.zeros([Q_HEAD_NUM], dtype=tl.float32)
    max_logic = tl.full([Q_HEAD_NUM], -float("inf"), dtype=tl.float32)
    acc = tl.zeros([Q_HEAD_NUM, BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        token_mask = offs_n_new < cur_batch_end_index
        k_loc = tl.load(
            Req_to_tokens
            + stride_req_to_tokens_b * cur_batch_req_idx
            + stride_req_to_tokens_s * offs_n_new,
            mask=token_mask,
            other=0,
        )
        if USE_64BIT_OFFSETS:
            k_loc = k_loc.to(tl.int64)
        off_k = (
            k_loc[None, :] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None] * stride_kd
        )
        k = tl.load(K + off_k, mask=token_mask[None, :] & dim_mask[:, None], other=0.0)
        raw_qk = tl.dot(q, k)
        raw_qk = tl.where(head_mask[:, None] & token_mask[None, :], raw_qk, -float("inf"))

        if STORE_SCORE_3D:
            off_score = (
                cur_batch * stride_asbs
                + cur_q_head_range[:, None] * stride_ash
                + offs_n_new[None, :] * stride_asl
            )
            tl.store(Attn_Score + off_score, raw_qk, mask=head_mask[:, None] & token_mask[None, :])
        if STORE_SCORE_2D:
            max_raw_qk = tl.max(raw_qk, axis=0)
            tl.atomic_max(
                Attn_Score + cur_batch * stride_asbs + offs_n_new * stride_asl,
                max_raw_qk,
                mask=token_mask,
            )

        logits = raw_qk * sm_scale
        off_v = (
            k_loc[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=token_mask[:, None] & dim_mask[None, :], other=0.0)
        cur_max_logic = tl.max(logits, axis=1)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        exp_logic = tl.exp(logits - new_max_logic[:, None])
        logic_scale = tl.exp(max_logic - new_max_logic)
        # Previous blocks use the old softmax origin and must be rescaled first.
        acc *= logic_scale[:, None]
        acc += tl.dot(exp_logic.to(v.dtype), v)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=1)
        max_logic = new_max_logic

    valid_block = block_n_size > 0
    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_q_head_range[:, None] * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d[None, :] * stride_mid_od
    )
    off_mid_lse = (
        cur_batch * stride_mid_o_eb
        + cur_q_head_range * stride_mid_o_eh
        + seq_start_block * stride_mid_o_es
    )
    tl.store(
        Mid_O + off_mid_o,
        acc / sum_exp[:, None],
        mask=valid_block & head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        Mid_O_LogExpSum + off_mid_lse,
        max_logic + tl.log(sum_exp),
        mask=valid_block & head_mask,
    )


@triton.jit
def _fwd_kernel_gqa_flash_decode_stage1_per_q(
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
    gqa_group_size,
    STORE_SCORE_3D: tl.constexpr,
    STORE_SCORE_2D: tl.constexpr,
    USE_64BIT_OFFSETS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    seq_start_block = tl.program_id(2)
    cur_kv_head = cur_head // gqa_group_size
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    cur_batch_start_index = seq_start_block * BLOCK_SEQ
    cur_batch_end_index = tl.minimum(cur_batch_seq_len, cur_batch_start_index + BLOCK_SEQ)
    block_n_size = (
        tl.where(
            cur_batch_end_index > cur_batch_start_index,
            cur_batch_end_index - cur_batch_start_index + BLOCK_N - 1,
            0,
        )
        // BLOCK_N
    )

    dim_mask = offs_d < HEAD_DIM
    q = tl.load(
        Q + cur_batch * stride_qbs + cur_head * stride_qh + offs_d * stride_qd,
        mask=dim_mask,
        other=0.0,
    )
    offs_n = cur_batch_start_index + tl.arange(0, BLOCK_N)
    sum_exp = 0.0
    max_logic = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    for start_n in range(0, block_n_size, 1):
        offs_n_new = start_n * BLOCK_N + offs_n
        token_mask = offs_n_new < cur_batch_end_index
        k_loc = tl.load(
            Req_to_tokens
            + stride_req_to_tokens_b * cur_batch_req_idx
            + stride_req_to_tokens_s * offs_n_new,
            mask=token_mask,
            other=0,
        )
        if USE_64BIT_OFFSETS:
            k_loc = k_loc.to(tl.int64)
        off_k = (
            k_loc[:, None] * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[None, :] * stride_kd
        )
        k = tl.load(K + off_k, mask=token_mask[:, None] & dim_mask[None, :], other=0.0)
        raw_qk = tl.sum(q[None, :].to(tl.float32) * k.to(tl.float32), axis=1)
        raw_qk = tl.where(token_mask, raw_qk, -float("inf"))
        if STORE_SCORE_3D:
            tl.store(
                Attn_Score + cur_batch * stride_asbs + cur_head * stride_ash + offs_n_new * stride_asl,
                raw_qk,
                mask=token_mask,
            )
        if STORE_SCORE_2D:
            tl.atomic_max(
                Attn_Score + cur_batch * stride_asbs + offs_n_new * stride_asl,
                raw_qk,
                mask=token_mask,
            )

        logits = raw_qk * sm_scale
        off_v = (
            k_loc[:, None] * stride_vbs
            + cur_kv_head * stride_vh
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(V + off_v, mask=token_mask[:, None] & dim_mask[None, :], other=0.0)
        cur_max_logic = tl.max(logits, axis=0)
        new_max_logic = tl.maximum(cur_max_logic, max_logic)
        exp_logic = tl.exp(logits - new_max_logic)
        logic_scale = tl.exp(max_logic - new_max_logic)
        acc *= logic_scale
        acc += tl.sum(exp_logic[:, None] * v.to(tl.float32), axis=0)
        sum_exp = sum_exp * logic_scale + tl.sum(exp_logic, axis=0)
        max_logic = new_max_logic

    valid_block = block_n_size > 0
    off_mid_o = (
        cur_batch * stride_mid_ob
        + cur_head * stride_mid_oh
        + seq_start_block * stride_mid_os
        + offs_d * stride_mid_od
    )
    off_mid_lse = (
        cur_batch * stride_mid_o_eb
        + cur_head * stride_mid_o_eh
        + seq_start_block * stride_mid_o_es
    )
    tl.store(Mid_O + off_mid_o, acc / sum_exp, mask=valid_block & dim_mask)
    tl.store(Mid_O_LogExpSum + off_mid_lse, max_logic + tl.log(sum_exp), mask=valid_block)


def _debug_validate_indices(req_to_tokens, b_req_idx, b_seqlen, max_len_in_batch, slot_capacity):
    if os.environ.get("SVLLM_DEBUG_DECODE_BOUNDS", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if b_req_idx.numel() == 0:
        raise ValueError("GQA decode requires at least one batch row.")
    min_req = int(b_req_idx.min().item())
    max_req = int(b_req_idx.max().item())
    if min_req < 0 or max_req >= req_to_tokens.shape[0]:
        raise ValueError(
            f"B_req_idx contains rows outside Req_to_tokens: min={min_req} max={max_req} "
            f"rows={req_to_tokens.shape[0]}."
        )
    min_len = int(b_seqlen.min().item())
    max_len = int(b_seqlen.max().item())
    if min_len <= 0 or max_len > max_len_in_batch or max_len > req_to_tokens.shape[1]:
        raise ValueError(
            f"B_Seqlen must be in [1, min(max_len_in_batch, token_width)], got "
            f"min={min_len} max={max_len} max_len_in_batch={max_len_in_batch} "
            f"token_width={req_to_tokens.shape[1]}."
        )
    rows = req_to_tokens.index_select(0, b_req_idx.to(torch.long))[:, :max_len]
    positions = torch.arange(max_len, device=rows.device)[None, :]
    valid = positions < b_seqlen[:, None]
    bad = ((rows < 0) | (rows >= slot_capacity)) & valid
    if bool(bad.any().item()):
        location = bad.nonzero(as_tuple=False)[0]
        batch_idx, token_idx = int(location[0].item()), int(location[1].item())
        slot = int(rows[batch_idx, token_idx].item())
        raise ValueError(
            f"Req_to_tokens slot index out of range: batch={batch_idx} token={token_idx} "
            f"slot={slot} capacity={slot_capacity}."
        )


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
    tensors = (q, k, v, req_to_tokens, b_req_idx, b_seqlen, mid_out, mid_out_logsumexp)
    if any(t.device.type != "cuda" for t in tensors):
        raise ValueError("GQA decode tensors must all be CUDA tensors.")
    if any(t.device != q.device for t in tensors):
        raise ValueError(f"GQA decode tensors must share one CUDA device, got {[str(t.device) for t in tensors]}.")
    if q.dim() != 3 or k.dim() != 3 or v.dim() != 3:
        raise ValueError(f"Q/K/V must be rank-3, got {q.shape}/{k.shape}/{v.shape}.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(
            f"Q/K/V head dimensions must match, got {q.shape[-1]}/{k.shape[-1]}/{v.shape[-1]}."
        )
    head_dim = int(q.shape[-1])
    if not (_MIN_HEAD_DIM <= head_dim <= _MAX_HEAD_DIM):
        raise ValueError(
            f"GQA decode head_dim must be in [{_MIN_HEAD_DIM}, {_MAX_HEAD_DIM}], got {head_dim}."
        )
    if q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"Q/K/V must share FP16 or BF16 dtype, got {q.dtype}/{k.dtype}/{v.dtype}.")
    if k.shape[0] != v.shape[0] or k.shape[1] != v.shape[1]:
        raise ValueError(f"K/V cache shapes are incompatible: {tuple(k.shape)} vs {tuple(v.shape)}.")
    q_heads, kv_heads = int(q.shape[1]), int(k.shape[1])
    if kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError(f"GQA requires q_heads divisible by kv_heads, got {q_heads}/{kv_heads}.")
    if q_heads <= kv_heads:
        raise ValueError(f"GQA requires q_heads > kv_heads, got {q_heads}/{kv_heads}.")
    if req_to_tokens.dim() != 2 or b_req_idx.dim() != 1 or b_seqlen.dim() != 1:
        raise ValueError(
            "Req_to_tokens must be rank-2 and B_req_idx/B_Seqlen rank-1, got "
            f"{req_to_tokens.shape}/{b_req_idx.shape}/{b_seqlen.shape}."
        )
    index_dtypes = {torch.int32, torch.int64}
    if any(t.dtype not in index_dtypes for t in (req_to_tokens, b_req_idx, b_seqlen)):
        raise ValueError(
            "Req_to_tokens, B_req_idx, and B_Seqlen must use int32 or int64, got "
            f"{req_to_tokens.dtype}/{b_req_idx.dtype}/{b_seqlen.dtype}."
        )
    batch = int(q.shape[0])
    if b_req_idx.numel() != batch or b_seqlen.numel() != batch:
        raise ValueError(
            f"Batch metadata must have {batch} rows, got {b_req_idx.numel()}/{b_seqlen.numel()}."
        )
    max_len_in_batch = int(max_len_in_batch)
    block_seq = int(block_seq)
    if max_len_in_batch <= 0:
        raise ValueError(f"max_len_in_batch must be positive, got {max_len_in_batch}.")
    if max_len_in_batch > req_to_tokens.shape[1]:
        raise ValueError(
            f"max_len_in_batch={max_len_in_batch} exceeds Req_to_tokens width={req_to_tokens.shape[1]}."
        )
    if block_seq <= 0 or block_seq % 16 != 0:
        raise ValueError(f"block_seq must be positive and divisible by 16, got {block_seq}.")
    required_blocks = triton.cdiv(max_len_in_batch, block_seq)
    if mid_out.dim() != 4 or tuple(mid_out.shape[:2]) != (batch, q_heads):
        raise ValueError(f"mid_out must have shape [B,Hq,blocks,D], got {tuple(mid_out.shape)}.")
    if mid_out.shape[2] < required_blocks or mid_out.shape[3] < head_dim or mid_out.dtype != torch.float32:
        raise ValueError(
            f"mid_out capacity/dtype is invalid: shape={tuple(mid_out.shape)} dtype={mid_out.dtype} "
            f"required_blocks={required_blocks} head_dim={head_dim}."
        )
    if mid_out_logsumexp.dim() != 3 or tuple(mid_out_logsumexp.shape[:2]) != (batch, q_heads):
        raise ValueError(f"mid_out_logsumexp must have shape [B,Hq,blocks], got {tuple(mid_out_logsumexp.shape)}.")
    if mid_out_logsumexp.shape[2] < required_blocks or mid_out_logsumexp.dtype != torch.float32:
        raise ValueError(
            f"mid_out_logsumexp capacity/dtype is invalid: shape={tuple(mid_out_logsumexp.shape)} "
            f"dtype={mid_out_logsumexp.dtype} required_blocks={required_blocks}."
        )
    if attn_score is not None:
        if attn_score.device != q.device or attn_score.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"attn_score must be a CUDA FP16/BF16/FP32 tensor, got {attn_score.device}/{attn_score.dtype}.")
        if attn_score.dim() == 2 and attn_score.dtype != torch.float32:
            raise ValueError(
                f"2D attn_score must be FP32 because Triton atomic_max does not support {attn_score.dtype}."
            )
        valid_shape = (
            attn_score.dim() == 3
            and attn_score.shape[0] == batch
            and attn_score.shape[1] == q_heads
            and attn_score.shape[2] >= max_len_in_batch
        ) or (
            attn_score.dim() == 2
            and attn_score.shape[0] == batch
            and attn_score.shape[1] >= max_len_in_batch
        )
        if not valid_shape:
            raise ValueError(
                f"attn_score must be [B,Hq,>=max_len] or [B,>=max_len], got {tuple(attn_score.shape)} "
                f"for B={batch} Hq={q_heads} max_len={max_len_in_batch}."
            )
    _debug_validate_indices(req_to_tokens, b_req_idx, b_seqlen, max_len_in_batch, int(k.shape[0]))


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
    *,
    attn_score,
    variant_id,
):
    _validate_inputs(
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
    config = get_stage1_variant_config(variant_id)
    schedule = config["schedule"]

    store_score_3d = attn_score is not None and attn_score.dim() == 3
    store_score_2d = attn_score is not None and attn_score.dim() == 2
    score = attn_score if attn_score is not None else mid_out
    stride_asbs = score.stride(0) if attn_score is not None else 0
    stride_ash = score.stride(1) if store_score_3d else 0
    stride_asl = score.stride(2) if store_score_3d else (score.stride(1) if store_score_2d else 0)
    group_size = q.shape[1] // k.shape[1]
    grid = (
        b_req_idx.shape[0],
        k.shape[1] if schedule == "grouped" else q.shape[1],
        triton.cdiv(max_len_in_batch, block_seq),
    )
    kernel = (
        _fwd_kernel_gqa_flash_decode_stage1_grouped
        if schedule == "grouped"
        else _fwd_kernel_gqa_flash_decode_stage1_per_q
    )
    meta = {
        "STORE_SCORE_3D": store_score_3d,
        "STORE_SCORE_2D": store_score_2d,
        "USE_64BIT_OFFSETS": _requires_64bit_cache_offsets(k, v),
        "BLOCK_SEQ": block_seq,
        "HEAD_DIM": q.shape[-1],
        "BLOCK_DMODEL": triton.next_power_of_2(q.shape[-1]),
        "BLOCK_N": config["block_n"],
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }
    if schedule == "grouped":
        meta["Q_HEAD_NUM"] = max(16, triton.next_power_of_2(group_size))
    kernel[grid](
        q,
        k,
        v,
        1.0 / (q.shape[-1] ** 0.5),
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
        group_size,
        **meta,
    )


@torch.no_grad()
def flash_decode_stage1_variant(
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
    *,
    variant_id,
    attn_score=None,
):
    """Research A/B entry point; production code uses the wrappers below."""
    _launch_stage1(
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
        attn_score=attn_score,
        variant_id=variant_id,
    )


_PRODUCTION_VARIANT_ID = "grouped_s1_bn16_w2_s2"


@torch.no_grad()
def flash_decode_stage1(
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
):
    _launch_stage1(
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
        variant_id=_PRODUCTION_VARIANT_ID,
    )


@torch.no_grad()
def flash_decode_stage1_with_score(
    q,
    k,
    v,
    req_to_tokens,
    b_req_idx,
    b_seqlen,
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
        req_to_tokens,
        b_req_idx,
        b_seqlen,
        max_len_in_batch,
        mid_out,
        mid_out_logsumexp,
        block_seq,
        attn_score=attn_score,
        variant_id=_PRODUCTION_VARIANT_ID,
    )
