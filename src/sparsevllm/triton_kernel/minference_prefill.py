from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import torch
import triton
import triton.language as tl

from sparsevllm.utils.profiler import profiler


MINFERENCE_LAST_Q = 64
MIN_VERTICAL_SIZE = 30
MIN_SLASH_SIZE = 50
MIN_SLASH_RECENT = 100
# The reference CUDA converter is fixed at 64x64. This custom paged-KV
# Triton path uses 128x128 on H100 to match svLLM's dense prefill tiling.
MINFERENCE_BLOCK_M = 128
MINFERENCE_BLOCK_N = 128
MINFERENCE_DENSE_FALLBACK_RATIO = 0.50
MINFERENCE_MIN_SPARSE_SEQ_LEN = 32768
MINFERENCE_TOPK_HEAD_CHUNK = 2


@lru_cache(maxsize=8)
def _load_pattern_config(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"MInference config must be a layer list: {path}")
    for layer_idx, layer in enumerate(data):
        if not isinstance(layer, dict):
            raise ValueError(f"MInference config layer {layer_idx} must be an object.")
    return data


def _get_vs_pattern(config, layer_idx: int, global_head_idx: int) -> tuple[int, int]:
    pattern_config = _load_pattern_config(str(config.minference_config_path))
    return _get_vs_pattern_from_config(pattern_config, config, layer_idx, global_head_idx)


def _get_vs_pattern_from_config(
    pattern_config: list[dict[str, Any]],
    config,
    layer_idx: int,
    global_head_idx: int,
) -> tuple[int, int]:
    if layer_idx >= len(pattern_config):
        raise ValueError(
            "MInference config has fewer layers than the model: "
            f"layer_idx={layer_idx}, config_layers={len(pattern_config)}."
        )
    layer_patterns = pattern_config[layer_idx]
    key = str(int(global_head_idx))
    if key not in layer_patterns:
        raise ValueError(
            "MInference config is missing a head pattern: "
            f"layer={layer_idx}, global_head={global_head_idx}."
        )
    pattern = layer_patterns[key]
    if not isinstance(pattern, (list, tuple)) or len(pattern) < 3:
        raise ValueError(
            "MInference head pattern must be a list like "
            "['vertical_and_slash', vertical_size, slash_size, score]. "
            f"layer={layer_idx}, global_head={global_head_idx}, value={pattern!r}."
        )
    pattern_type = str(pattern[0])
    if pattern_type != "vertical_and_slash":
        raise NotImplementedError(
            "MInference prefill V1 supports only vertical_and_slash patterns, "
            f"got {pattern_type!r} at layer={layer_idx}, global_head={global_head_idx}."
        )
    ratio = float(config.minference_ratio)
    vertical_size = int(int(pattern[1]) * ratio)
    slash_size = int(int(pattern[2]) * ratio)
    return vertical_size, slash_size


def _estimate_layer_pattern_density(
    config,
    layer_idx: int,
    rank: int,
    num_heads: int,
    seq_len: int,
) -> float:
    pattern_config = _load_pattern_config(str(config.minference_config_path))
    if seq_len <= 0:
        return 1.0

    total = 0
    for head_idx in range(num_heads):
        global_head_idx = int(rank) * num_heads + head_idx
        vertical_size, slash_size = _get_vs_pattern_from_config(
            pattern_config, config, layer_idx, global_head_idx
        )
        vertical_size = min(seq_len, max(int(vertical_size), MIN_VERTICAL_SIZE))
        slash_size = min(seq_len, max(int(slash_size), MIN_SLASH_SIZE))
        total += vertical_size + slash_size
    return float(total) / float(num_heads * seq_len)


def _get_layer_pattern_sizes(
    pattern_config: list[dict[str, Any]],
    config,
    layer_idx: int,
    rank: int,
    num_heads: int,
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    vertical_sizes = []
    slash_sizes = []
    for head_idx in range(num_heads):
        global_head_idx = int(rank) * num_heads + head_idx
        vertical_size, slash_size = _get_vs_pattern_from_config(
            pattern_config, config, layer_idx, global_head_idx
        )
        vertical_sizes.append(min(seq_len, max(int(vertical_size), MIN_VERTICAL_SIZE)))
        slash_sizes.append(min(seq_len, max(int(slash_size), MIN_SLASH_SIZE)))
    return (
        torch.tensor(vertical_sizes, dtype=torch.int64, device=device),
        torch.tensor(slash_sizes, dtype=torch.int64, device=device),
    )


def _sum_all_diagonal_matrix(mat: torch.Tensor) -> torch.Tensor:
    # Same diagonal aggregation as MInference's sum_all_diagonal_matrix,
    # specialized to [heads, last_q, seq_len].
    heads, rows, cols = mat.shape
    padded = torch.nn.functional.pad(mat, (rows, rows), "constant", 0.0)
    padded_cols = cols + 2 * rows
    strided = padded.as_strided(
        (heads, rows, rows + cols),
        (rows * padded_cols, padded_cols + 1, 1),
    )
    return torch.sum(strided, dim=1)[:, 1:].contiguous()


def _build_batch_head_indices(
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    vertical_sizes: torch.Tensor,
    slash_sizes: torch.Tensor,
    kv_group_num: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = int(q_seq.shape[0])
    num_heads = int(q_seq.shape[1])
    head_dim = int(q_seq.shape[-1])
    last_q = min(MINFERENCE_LAST_Q, seq_len)
    max_vertical = int(vertical_sizes.max().item())
    max_slash = int(slash_sizes.max().item())
    q_positions = torch.arange(seq_len - last_q, seq_len, device=q_seq.device)
    k_positions = torch.arange(seq_len, device=q_seq.device)

    vertical_chunks = []
    slash_chunks = []
    for head_start in range(0, num_heads, MINFERENCE_TOPK_HEAD_CHUNK):
        head_end = min(num_heads, head_start + MINFERENCE_TOPK_HEAD_CHUNK)
        head_ids = torch.arange(head_start, head_end, device=q_seq.device)
        kv_head_ids = head_ids // int(kv_group_num)
        q_last = q_seq[-last_q:, head_start:head_end, :].permute(1, 0, 2).to(torch.float32)
        k_by_q_head = k_seq.index_select(1, kv_head_ids).permute(1, 2, 0).to(torch.float32)
        logits = torch.bmm(q_last, k_by_q_head) / math.sqrt(head_dim)
        logits = logits.masked_fill(k_positions.view(1, 1, -1) > q_positions.view(1, -1, 1), -torch.inf)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

        vertical_scores = probs.sum(dim=1)
        vertical_scores[:, : min(MIN_VERTICAL_SIZE, seq_len)] = torch.inf
        vertical_chunks.append(torch.topk(vertical_scores, max_vertical, dim=-1).indices.to(torch.int32))

        slash_scores = _sum_all_diagonal_matrix(probs)[:, :seq_len]
        slash_scores[:, -min(MIN_SLASH_RECENT, seq_len):] = torch.inf
        slash_topk = torch.topk(slash_scores, max_slash, dim=-1).indices
        slash_chunks.append(((seq_len - 1) - slash_topk).to(torch.int32))

    vertical_idx = torch.cat(vertical_chunks, dim=0)
    slash_idx = torch.cat(slash_chunks, dim=0)
    return vertical_idx, vertical_sizes.to(torch.int32), slash_idx, slash_sizes.to(torch.int32)


def _save_block_offsets(range_start: int, range_end: int, block_n: int) -> list[int]:
    return list(range(int(range_start), int(range_end), int(block_n)))


def _convert_vertical_slash_row(
    vertical_list: list[int],
    slash_list: list[int],
    *,
    end_m: int,
    block_m: int,
    block_n: int,
) -> tuple[list[int], list[int]]:
    # Mirrors MInference vertical/slash range merging. The no-slash fallback
    # follows the older torch/pycuda builder to keep only visible verticals.
    if not slash_list:
        return [], [idx for idx in vertical_list if idx < end_m]

    block_offsets: list[int] = []
    column_indexes: list[int] = []
    s = 0
    v = 0

    while s < len(slash_list) and slash_list[s] >= end_m:
        s += 1
    if s >= len(slash_list):
        return [], [idx for idx in vertical_list if idx < end_m]

    range_end = max(end_m - int(slash_list[s]), block_m)
    range_start = range_end - block_m
    s += 1

    v_idx = vertical_list[v] if v < len(vertical_list) else end_m + block_m
    while True:
        if v_idx < range_end:
            if v_idx < range_start:
                column_indexes.append(v_idx)
            v += 1
            v_idx = vertical_list[v] if v < len(vertical_list) else end_m + block_m
            continue

        if s >= len(slash_list):
            block_offsets.extend(_save_block_offsets(range_start, range_end, block_n))
            break

        next_range_end = max(end_m - int(slash_list[s]), block_m)
        s += 1
        if next_range_end > range_end + block_m:
            block_offsets.extend(_save_block_offsets(range_start, range_end, block_n))
            range_start = next_range_end - block_m
            range_end = next_range_end
        elif next_range_end > range_end:
            range_end += block_m

    return block_offsets, column_indexes


@triton.jit
def _convert_vertical_slash_indexes_kernel(
    B_Seqlen,
    Vertical_Indexes,
    Slash_Indexes,
    Vertical_Counts,
    Slash_Counts,
    Block_Count,
    Block_Offset,
    Column_Count,
    Column_Index,
    stride_vi_b, stride_vi_h, stride_vi_n,
    stride_si_b, stride_si_h, stride_si_n,
    stride_vc_b, stride_vc_h,
    stride_sc_b, stride_sc_h,
    stride_bc_b, stride_bc_h, stride_bc_m,
    stride_bo_b, stride_bo_h, stride_bo_m, stride_bo_n,
    stride_cc_b, stride_cc_h, stride_cc_m,
    stride_ci_b, stride_ci_h, stride_ci_m, stride_ci_n,
    H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    start_m = row_idx * BLOCK_M
    seq_len = tl.load(B_Seqlen + cur_batch)
    if start_m >= seq_len:
        return

    end_m = start_m + BLOCK_M
    vertical_count = tl.load(Vertical_Counts + cur_batch * stride_vc_b + cur_head * stride_vc_h)
    slash_count = tl.load(Slash_Counts + cur_batch * stride_sc_b + cur_head * stride_sc_h)

    vi_base = Vertical_Indexes + cur_batch * stride_vi_b + cur_head * stride_vi_h
    si_base = Slash_Indexes + cur_batch * stride_si_b + cur_head * stride_si_h
    bo_base = Block_Offset + cur_batch * stride_bo_b + cur_head * stride_bo_h + row_idx * stride_bo_m
    ci_base = Column_Index + cur_batch * stride_ci_b + cur_head * stride_ci_h + row_idx * stride_ci_m

    tmp_blk_cnt = tl.full((), 0, tl.int32)
    tmp_col_cnt = tl.full((), 0, tl.int32)
    s = tl.full((), 0, tl.int32)
    v = tl.full((), 0, tl.int32)

    found_slash = tl.full((), 0, tl.int32)
    while (s < slash_count) & (found_slash == 0):
        slash_value = tl.load(si_base + s * stride_si_n)
        if slash_value < end_m:
            found_slash = 1
        else:
            s += 1
    if s >= slash_count:
        while v < vertical_count:
            v_idx = tl.load(vi_base + v * stride_vi_n)
            if v_idx < end_m:
                tl.store(ci_base + tmp_col_cnt * stride_ci_n, v_idx)
                tmp_col_cnt += 1
            v += 1
        tl.store(Column_Count + cur_batch * stride_cc_b + cur_head * stride_cc_h + row_idx * stride_cc_m, tmp_col_cnt)
        return

    slash_value = tl.load(si_base + s * stride_si_n)
    range_end = tl.maximum(end_m - slash_value, BLOCK_M)
    range_start = range_end - BLOCK_M
    s += 1
    v_idx = tl.full((), end_m + BLOCK_M, tl.int32)
    if v < vertical_count:
        v_idx = tl.load(vi_base + v * stride_vi_n)

    active = tl.full((), 1, tl.int32)
    while active == 1:
        if v_idx < range_end:
            if v_idx < range_start:
                tl.store(ci_base + tmp_col_cnt * stride_ci_n, v_idx)
                tmp_col_cnt += 1
            v += 1
            v_idx = end_m + BLOCK_M
            if v < vertical_count:
                v_idx = tl.load(vi_base + v * stride_vi_n)
        else:
            if s >= slash_count:
                offset = range_start
                while offset < range_end:
                    tl.store(bo_base + tmp_blk_cnt * stride_bo_n, offset)
                    tmp_blk_cnt += 1
                    offset += BLOCK_N
                active = 0
            else:
                slash_value = tl.load(si_base + s * stride_si_n)
                next_range_end = tl.maximum(end_m - slash_value, BLOCK_M)
                s += 1
                if next_range_end > range_end + BLOCK_M:
                    offset = range_start
                    while offset < range_end:
                        tl.store(bo_base + tmp_blk_cnt * stride_bo_n, offset)
                        tmp_blk_cnt += 1
                        offset += BLOCK_N
                    range_start = next_range_end - BLOCK_M
                    range_end = next_range_end
                elif next_range_end > range_end:
                    range_end += BLOCK_M

    tl.store(Block_Count + cur_batch * stride_bc_b + cur_head * stride_bc_h + row_idx * stride_bc_m, tmp_blk_cnt)
    tl.store(Column_Count + cur_batch * stride_cc_b + cur_head * stride_cc_h + row_idx * stride_cc_m, tmp_col_cnt)


@torch.no_grad()
def _build_sparse_metadata(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    req_to_tokens: torch.Tensor,
    *,
    layer_idx: int,
    config,
    rank: int,
    block_m: int,
    block_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = int(b_seq_len.numel())
    num_heads = int(q.shape[1])
    kv_group_num = num_heads // int(k_cache.shape[1])
    max_seq_len = int(b_seq_len.max().item())
    num_rows = triton.cdiv(max_seq_len, block_m)

    pattern_config = _load_pattern_config(str(config.minference_config_path))
    vertical_sizes, slash_sizes = _get_layer_pattern_sizes(
        pattern_config,
        config,
        layer_idx,
        rank,
        num_heads,
        max_seq_len,
        q.device,
    )
    max_vertical = int(vertical_sizes.max().item())
    max_slash = int(slash_sizes.max().item())
    vertical_indexes = torch.zeros((batch, num_heads, max_vertical), dtype=torch.int32, device=q.device)
    slash_indexes = torch.zeros((batch, num_heads, max_slash), dtype=torch.int32, device=q.device)
    vertical_counts = torch.zeros((batch, num_heads), dtype=torch.int32, device=q.device)
    slash_counts = torch.zeros((batch, num_heads), dtype=torch.int32, device=q.device)

    for b_idx in range(batch):
        seq_len = int(b_seq_len[b_idx].item())
        q_start = int(b_start_loc[b_idx].item())
        req_row = int(b_req_idx[b_idx].item())
        slots = req_to_tokens[req_row, :seq_len].to(torch.long)
        q_seq = q[q_start: q_start + seq_len]
        k_seq = k_cache.index_select(0, slots)
        with profiler.record("minference_topk"):
            batch_vertical_idx, batch_vertical_sizes, batch_slash_idx, batch_slash_sizes = _build_batch_head_indices(
                q_seq,
                k_seq,
                vertical_sizes.clamp_max(seq_len),
                slash_sizes.clamp_max(seq_len),
                kv_group_num,
            )
        vertical_counts[b_idx] = batch_vertical_sizes
        slash_counts[b_idx] = batch_slash_sizes
        for head_idx in range(num_heads):
            vertical_count = int(batch_vertical_sizes[head_idx].item())
            slash_count = int(batch_slash_sizes[head_idx].item())
            vertical_indexes[b_idx, head_idx, :vertical_count] = torch.sort(
                batch_vertical_idx[head_idx, :vertical_count],
                dim=-1,
            ).values
            slash_indexes[b_idx, head_idx, :slash_count] = torch.sort(
                batch_slash_idx[head_idx, :slash_count],
                dim=-1,
                descending=True,
            ).values

    block_count = torch.zeros((batch, num_heads, num_rows), dtype=torch.int32, device=q.device)
    block_offset = torch.zeros((batch, num_heads, num_rows, max_slash), dtype=torch.int32, device=q.device)
    column_count = torch.zeros((batch, num_heads, num_rows), dtype=torch.int32, device=q.device)
    column_index = torch.zeros((batch, num_heads, num_rows, max_vertical), dtype=torch.int32, device=q.device)

    grid = (num_rows, batch * num_heads)
    with profiler.record("minference_metadata_convert"):
        _convert_vertical_slash_indexes_kernel[grid](
            b_seq_len,
            vertical_indexes,
            slash_indexes,
            vertical_counts,
            slash_counts,
            block_count,
            block_offset,
            column_count,
            column_index,
            vertical_indexes.stride(0),
            vertical_indexes.stride(1),
            vertical_indexes.stride(2),
            slash_indexes.stride(0),
            slash_indexes.stride(1),
            slash_indexes.stride(2),
            vertical_counts.stride(0),
            vertical_counts.stride(1),
            slash_counts.stride(0),
            slash_counts.stride(1),
            block_count.stride(0),
            block_count.stride(1),
            block_count.stride(2),
            block_offset.stride(0),
            block_offset.stride(1),
            block_offset.stride(2),
            block_offset.stride(3),
            column_count.stride(0),
            column_count.stride(1),
            column_count.stride(2),
            column_index.stride(0),
            column_index.stride(1),
            column_index.stride(2),
            column_index.stride(3),
            H=num_heads,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=1,
        )
    return block_count, block_offset, column_count, column_index


@triton.jit
def _minference_prefill_kernel(
    Q, K, V, sm_scale, Out,
    B_Start_Loc, B_Seqlen, Req_to_tokens, B_req_idx,
    Block_Count, Block_Offset, Column_Count, Column_Index,
    Attn_Score,
    stride_qbs, stride_qh, stride_qd,
    stride_kbs, stride_kh, stride_kd,
    stride_vbs, stride_vh, stride_vd,
    stride_obs, stride_oh, stride_od,
    stride_req_to_tokens_b, stride_req_to_tokens_s,
    stride_bc_b, stride_bc_h, stride_bc_m,
    stride_bo_b, stride_bo_h, stride_bo_m, stride_bo_n,
    stride_cc_b, stride_cc_h, stride_cc_m,
    stride_ci_b, stride_ci_h, stride_ci_m, stride_ci_n,
    stride_asb, stride_ash, stride_asl,
    kv_group_num,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_SCORE: tl.constexpr,
):
    start_m = tl.program_id(0)
    cur_bh = tl.program_id(1)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    seq_start = tl.load(B_Start_Loc + cur_batch)
    seq_len = tl.load(B_Seqlen + cur_batch)
    if start_m * BLOCK_M >= seq_len:
        return
    req_idx = tl.load(B_req_idx + cur_batch)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    m_mask = offs_m < seq_len

    q_ptrs = (
        Q + (seq_start + offs_m[:, None]) * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    block_count = tl.load(
        Block_Count
        + cur_batch * stride_bc_b
        + cur_head * stride_bc_h
        + start_m * stride_bc_m
    )
    block_offsets = (
        Block_Offset
        + cur_batch * stride_bo_b
        + cur_head * stride_bo_h
        + start_m * stride_bo_m
    )
    for block_idx in range(block_count):
        start_n = tl.load(block_offsets + block_idx * stride_bo_n)
        cols = start_n + offs_n
        n_mask = cols < seq_len
        kv_loc = tl.load(
            Req_to_tokens + req_idx * stride_req_to_tokens_b + cols * stride_req_to_tokens_s,
            mask=n_mask,
            other=0,
        )
        k_ptrs = K + kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        v_ptrs = V + kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k)
        causal_mask = cols[None, :] <= offs_m[:, None]
        score_mask = m_mask[:, None] & n_mask[None, :] & causal_mask
        if HAS_SCORE:
            score_to_collect = tl.where(score_mask, qk, 0.0)
            tl.atomic_add(
                Attn_Score
                + cur_batch * stride_asb
                + cur_head * stride_ash
                + cols * stride_asl,
                tl.sum(score_to_collect, 0),
                mask=n_mask,
            )
        qk = tl.where(score_mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    column_count = tl.load(
        Column_Count
        + cur_batch * stride_cc_b
        + cur_head * stride_cc_h
        + start_m * stride_cc_m
    )
    column_index = (
        Column_Index
        + cur_batch * stride_ci_b
        + cur_head * stride_ci_h
        + start_m * stride_ci_m
    )
    for column_start in range(0, column_count, BLOCK_N):
        col_offsets = column_start + offs_n
        n_mask = col_offsets < column_count
        cols = tl.load(column_index + col_offsets * stride_ci_n, mask=n_mask, other=0)
        n_mask = n_mask & (cols < seq_len)
        kv_loc = tl.load(
            Req_to_tokens + req_idx * stride_req_to_tokens_b + cols * stride_req_to_tokens_s,
            mask=n_mask,
            other=0,
        )
        k_ptrs = K + kv_loc[None, :] * stride_kbs + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        v_ptrs = V + kv_loc[:, None] * stride_vbs + cur_kv_head * stride_vh + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0)
        qk = tl.dot(q, k)
        causal_mask = cols[None, :] <= offs_m[:, None]
        score_mask = m_mask[:, None] & n_mask[None, :] & causal_mask
        if HAS_SCORE:
            score_to_collect = tl.where(score_mask, qk, 0.0)
            tl.atomic_add(
                Attn_Score
                + cur_batch * stride_asb
                + cur_head * stride_ash
                + cols * stride_asl,
                tl.sum(score_to_collect, 0),
                mask=n_mask,
            )
        qk = tl.where(score_mask, qk * sm_scale, -1.0e8)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out + (seq_start + offs_m[:, None]) * stride_obs
        + cur_head * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc, mask=m_mask[:, None])


@torch.no_grad()
def minference_context_attention_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    o: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_input_len: int,
    req_to_tokens: torch.Tensor,
    *,
    layer_idx: int,
    config,
    rank: int,
    attn_score: torch.Tensor | None = None,
):
    if b_prompt_cache_len.numel() and bool(torch.any(b_prompt_cache_len != 0).item()):
        raise RuntimeError(
            "MInference prefill does not support chunk/prefix prefill yet. "
            "Increase engine_prefill_chunk_size so each prompt is prefetched in one step."
        )
    if attn_score is not None and attn_score.dim() != 3:
        raise NotImplementedError("MInference prefill currently supports only 3D attention-score collection.")

    block_m = MINFERENCE_BLOCK_M
    block_n = MINFERENCE_BLOCK_N
    head_dim = int(q.shape[-1])
    if head_dim not in {16, 32, 64, 128, 256}:
        raise ValueError(f"Unsupported MInference head_dim={head_dim}.")
    if q.dtype != k_cache.dtype or k_cache.dtype != v_cache.dtype:
        raise ValueError("MInference prefill requires q, k_cache, and v_cache to have the same dtype.")
    if not (q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1 and o.stride(-1) == 1):
        raise ValueError("MInference prefill expects contiguous head_dim strides.")

    max_seq_len = int(b_seq_len.max().item())
    density = _estimate_layer_pattern_density(config, layer_idx, rank, int(q.shape[1]), max_seq_len)
    if max_seq_len < MINFERENCE_MIN_SPARSE_SEQ_LEN or density >= MINFERENCE_DENSE_FALLBACK_RATIO:
        from sparsevllm.triton_kernel.context_flashattention_nopad import context_attention_fwd

        context_attention_fwd(
            q,
            k_cache,
            v_cache,
            o,
            b_req_idx,
            b_start_loc,
            b_seq_len,
            b_prompt_cache_len,
            max_input_len,
            req_to_tokens,
            attn_score=attn_score,
        )
        return

    block_count, block_offset, column_count, column_index = _build_sparse_metadata(
        q,
        k_cache,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        req_to_tokens,
        layer_idx=layer_idx,
        config=config,
        rank=rank,
        block_m=block_m,
        block_n=block_n,
    )

    sm_scale = 1.0 / math.sqrt(head_dim) * 1.4426950408889634
    batch = int(b_seq_len.shape[0])
    num_heads = int(q.shape[1])
    kv_group_num = num_heads // int(k_cache.shape[1])
    grid = (int(block_count.shape[2]), batch * num_heads, 1)
    score_arg = attn_score if attn_score is not None else o
    with profiler.record("minference_sparse_kernel"):
        _minference_prefill_kernel[grid](
            q,
            k_cache,
            v_cache,
            sm_scale,
            o,
            b_start_loc,
            b_seq_len,
            req_to_tokens,
            b_req_idx,
            block_count,
            block_offset,
            column_count,
            column_index,
            score_arg,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            req_to_tokens.stride(0),
            req_to_tokens.stride(1),
            block_count.stride(0),
            block_count.stride(1),
            block_count.stride(2),
            block_offset.stride(0),
            block_offset.stride(1),
            block_offset.stride(2),
            block_offset.stride(3),
            column_count.stride(0),
            column_count.stride(1),
            column_count.stride(2),
            column_index.stride(0),
            column_index.stride(1),
            column_index.stride(2),
            column_index.stride(3),
            score_arg.stride(0),
            score_arg.stride(1) if attn_score is not None else 0,
            score_arg.stride(2) if attn_score is not None else 0,
            kv_group_num=kv_group_num,
            H=num_heads,
            BLOCK_DMODEL=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            HAS_SCORE=attn_score is not None,
            num_warps=4 if head_dim <= 64 else 8,
            num_stages=2,
        )
