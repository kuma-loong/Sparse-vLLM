import torch
import triton
import triton.language as tl


PREFILL_SCORE_VARIANTS = (
    "three_pass_current",
    "three_pass_host_bounds",
    "three_pass_host_bounds_bh2",
    "three_pass_bn32",
    "three_pass_bn64",
    "three_pass_bn128",
    "three_pass_bn256",
    "three_pass_bh1",
    "three_pass_bh2",
    "three_pass_bh4",
    "three_pass_bh8",
    "three_pass_warps4",
    "three_pass_warps8",
    "three_pass_stages1",
    "three_pass_stages2",
    "three_pass_stages3",
    "three_pass_stages4",
    "three_pass_reduce_rows1",
    "three_pass_reduce_rows4",
    "three_pass_reduce_rows8",
    "three_pass_reduce_rows16",
    "three_pass_reduce_rows32",
)


def get_prefill_score_variant_config(
    variant_id: str,
    *,
    head_dim: int,
    max_score_len: int,
    kv_group_num: int,
    candidate_blocks: int,
) -> dict[str, int | bool]:
    if variant_id not in PREFILL_SCORE_VARIANTS:
        raise ValueError(f"unknown prefill score variant: {variant_id!r}")

    block_m = max(16, triton.next_power_of_2(max_score_len))
    block_n = 64 if head_dim >= 128 else 128
    block_h_limit = max(1, min(8, 256 // block_m))
    block_h = min(triton.next_power_of_2(kv_group_num), block_h_limit)
    dot_stages = 3
    reduce_rows = 16

    if variant_id == "three_pass_host_bounds_bh2":
        if head_dim == 128:
            block_h = min(2, triton.next_power_of_2(kv_group_num))
    elif variant_id.startswith("three_pass_bn"):
        block_n = int(variant_id.removeprefix("three_pass_bn"))
    elif variant_id.startswith("three_pass_bh"):
        requested_block_h = int(variant_id.removeprefix("three_pass_bh"))
        block_h = min(requested_block_h, triton.next_power_of_2(kv_group_num))
    elif variant_id.startswith("three_pass_stages"):
        dot_stages = int(variant_id.removeprefix("three_pass_stages"))
    elif variant_id.startswith("three_pass_reduce_rows"):
        reduce_rows = int(variant_id.removeprefix("three_pass_reduce_rows"))

    block_rows = block_h * block_m
    dot_warps = 8 if block_rows >= 128 or block_n >= 128 else 4
    if variant_id == "three_pass_warps4":
        dot_warps = 4
    elif variant_id == "three_pass_warps8":
        dot_warps = 8

    reduce_blocks = triton.next_power_of_2(candidate_blocks)
    while reduce_rows > 1 and reduce_rows * reduce_blocks > 32768:
        reduce_rows //= 2
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_h": block_h,
        "block_rows": block_rows,
        "dot_warps": dot_warps,
        "dot_stages": dot_stages,
        "reduce_blocks": reduce_blocks,
        "reduce_rows": reduce_rows,
        "reduce_warps": 8 if reduce_blocks >= 1024 else 4,
        "reduce_stages": 4,
        "use_host_bounds": variant_id in {
            "three_pass_host_bounds",
            "three_pass_host_bounds_bh2",
        },
    }


@triton.jit
def _prefill_score_partial_stats_kernel(
    Q,
    K,
    Partial_M,
    Partial_L,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    candidate_start: tl.constexpr,
    num_recent_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = score_q_start + (offs_rows % BLOCK_M)
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - num_recent_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    if USE_IEEE:
        qk = tl.dot(q, k, input_precision="ieee") * sm_scale
    else:
        qk = tl.dot(q, k) * sm_scale
    causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
    valid = q_row_valid[:, None] & kv_in_candidate[None, :] & causal_mask
    qk = tl.where(valid, qk, -1.0e20)
    m_i = tl.max(qk, axis=1)
    p = tl.exp(qk - m_i[:, None])
    p = tl.where(valid, p, 0.0)
    l_i = tl.sum(p, axis=1)

    stats_offs = (cur_group * NUM_BLOCKS + cur_n_block) * BLOCK_ROWS + offs_rows
    tl.store(Partial_M + stats_offs, m_i)
    tl.store(Partial_L + stats_offs, l_i)


@triton.jit
def _prefill_score_reduce_stats_kernel(
    Partial_M,
    Partial_L,
    Global_M,
    Global_L,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    REDUCE_BLOCKS: tl.constexpr,
    REDUCE_ROWS: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_row_block = tl.program_id(1)
    offs_rows = cur_row_block * REDUCE_ROWS + tl.arange(0, REDUCE_ROWS)
    offs_blocks = tl.arange(0, REDUCE_BLOCKS)

    stats_offs = (
        cur_group * NUM_BLOCKS * BLOCK_ROWS
        + offs_blocks[None, :] * BLOCK_ROWS
        + offs_rows[:, None]
    )
    mask = (offs_rows[:, None] < BLOCK_ROWS) & (offs_blocks[None, :] < NUM_BLOCKS)
    partial_m = tl.load(Partial_M + stats_offs, mask=mask, other=-1.0e20)
    partial_l = tl.load(Partial_L + stats_offs, mask=mask, other=0.0)
    m_i = tl.max(partial_m, axis=1)
    l_i = tl.sum(partial_l * tl.exp(partial_m - m_i[:, None]), axis=1)

    out_offs = cur_group * BLOCK_ROWS + offs_rows
    tl.store(Global_M + out_offs, m_i, mask=offs_rows < BLOCK_ROWS)
    tl.store(Global_L + out_offs, l_i, mask=offs_rows < BLOCK_ROWS)


@triton.jit
def _prefill_score_final_kernel(
    Q,
    K,
    Attn_Score,
    Global_M,
    Global_L,
    B_Seqlen,
    Req_to_tokens,
    B_req_idx,
    Score_Q_Start,
    Score_Q_End,
    B_Start_Loc,
    B_Prompt_Cache_Len,
    stride_qt,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_asb,
    stride_asl,
    stride_req_to_tokens_b,
    stride_req_to_tokens_s,
    H_PER_KV: tl.constexpr,
    H_KV: tl.constexpr,
    HEAD_BLOCKS: tl.constexpr,
    candidate_start: tl.constexpr,
    num_recent_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_IEEE: tl.constexpr,
):
    cur_group = tl.program_id(0)
    cur_n_block = tl.program_id(1)
    cur_head_block = cur_group % HEAD_BLOCKS
    cur_bkv = cur_group // HEAD_BLOCKS
    cur_batch = cur_bkv // H_KV
    cur_kv_head = cur_bkv % H_KV

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = context_len - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)
    score_q_len = tl.maximum(score_q_end - score_q_start, 1)

    offs_rows = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_n = tl.arange(0, BLOCK_N)
    local_head = cur_head_block * BLOCK_H + offs_rows // BLOCK_M
    q_head = cur_kv_head * H_PER_KV + local_head
    q_abs_pos = score_q_start + (offs_rows % BLOCK_M)
    q_rel_pos = q_abs_pos - prompt_cache_len
    row_head_in_block = offs_rows // BLOCK_M
    q_row_valid = (
        (local_head < H_PER_KV)
        & (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + q_head[:, None] * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_row_valid[:, None], other=0.0)

    start_n = cur_n_block * BLOCK_N
    kv_pos = start_n + offs_n
    candidate_end = tl.maximum(candidate_start, context_len - num_recent_tokens)
    kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
    kv_loc = tl.load(
        Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
        mask=kv_in_candidate,
        other=0,
    )
    off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
    k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)

    if USE_IEEE:
        qk = tl.dot(q, k, input_precision="ieee") * sm_scale
    else:
        qk = tl.dot(q, k) * sm_scale
    causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
    valid = q_row_valid[:, None] & kv_in_candidate[None, :] & causal_mask
    qk = tl.where(valid, qk, -1.0e20)

    stats_offs = cur_group * BLOCK_ROWS + offs_rows
    m_i = tl.load(Global_M + stats_offs)
    l_i = tl.load(Global_L + stats_offs)
    safe_l_i = tl.where(l_i > 0.0, l_i, 1.0)
    probs = tl.exp(qk - m_i[:, None]) / safe_l_i[:, None]
    probs = tl.where(valid, probs, 0.0)

    token_score = tl.zeros([BLOCK_N], dtype=tl.float32)
    for head_idx in tl.static_range(0, BLOCK_H):
        head_rows = row_head_in_block == head_idx
        head_score = tl.sum(tl.where(head_rows[:, None], probs, 0.0), axis=0) / (score_q_len * 1.0)
        token_score = tl.maximum(token_score, head_score)

    tl.atomic_max(
        Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
        token_score,
        mask=kv_in_candidate,
    )


@torch.no_grad()
def prefill_score_fwd_variant(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_score: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_query_len: int,
    req_to_token_indexs: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    candidate_start: int = 0,
    num_recent_tokens: int = 0,
    variant_id: str = "three_pass_current",
    stage: str = "combined",
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    host_max_score_len: int | None = None,
    host_max_candidate_end: int | None = None,
    use_provided_bounds: bool = False,
):
    head_dim = q.shape[-1]
    assert k.shape[-1] == head_dim
    assert q.dtype == k.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert attn_score.dim() == 2
    if attn_score.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"prefill score output must be fp16, bf16, or fp32, got {attn_score.dtype}")
    assert head_dim in {16, 32, 64, 128, 256}
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_head = k.shape[1]
    kv_group_num = head // kv_head
    if kv_group_num <= 0 or head % kv_head != 0:
        raise ValueError(f"num query heads must be divisible by num kv heads: q={head} k={kv_head}")
    if stage not in {"partial", "reduce", "final", "combined"}:
        raise ValueError(f"unknown prefill score stage: {stage!r}")
    use_host_bounds = variant_id in {
        "three_pass_host_bounds",
        "three_pass_host_bounds_bh2",
    } or use_provided_bounds
    if use_host_bounds:
        if host_max_score_len is None or host_max_candidate_end is None:
            raise ValueError("three_pass_host_bounds requires both host bounds")
        max_score_len = int(host_max_score_len)
    else:
        max_score_len = int((score_q_end - score_q_start).max().item())
    if max_score_len <= 0:
        return None
    if max_score_len > 128:
        raise ValueError(f"prefill score query range is too large for this kernel: {max_score_len} > 128")

    if use_host_bounds:
        max_candidate_end = max(int(candidate_start), int(host_max_candidate_end))
    else:
        candidate_ends = torch.clamp(b_seq_len - int(num_recent_tokens), min=int(candidate_start))
        max_candidate_end = int(candidate_ends.max().item()) if batch > 0 else 0
    if max_candidate_end <= int(candidate_start):
        return None

    initial_block_n = 64 if head_dim >= 128 else 128
    initial_candidate_blocks = triton.cdiv(max_candidate_end, initial_block_n)
    initial_config = get_prefill_score_variant_config(
        variant_id,
        head_dim=head_dim,
        max_score_len=max_score_len,
        kv_group_num=kv_group_num,
        candidate_blocks=initial_candidate_blocks,
    )
    block_m = int(initial_config["block_m"])
    block_n = int(initial_config["block_n"])
    candidate_blocks = triton.cdiv(max_candidate_end, block_n)
    if candidate_blocks <= 0:
        return None

    config = get_prefill_score_variant_config(
        variant_id,
        head_dim=head_dim,
        max_score_len=max_score_len,
        kv_group_num=kv_group_num,
        candidate_blocks=candidate_blocks,
    )
    block_h = int(config["block_h"])
    head_blocks = triton.cdiv(kv_group_num, block_h)
    block_rows = int(config["block_rows"])
    group_count = batch * kv_head * head_blocks
    if group_count <= 0:
        return None

    expected_shapes = (
        (group_count, candidate_blocks, block_rows),
        (group_count, candidate_blocks, block_rows),
        (group_count, block_rows),
        (group_count, block_rows),
    )
    if workspace is None:
        partial_m = torch.empty(expected_shapes[0], device=q.device, dtype=torch.float32)
        partial_l = torch.empty_like(partial_m)
        global_m = torch.empty(expected_shapes[2], device=q.device, dtype=torch.float32)
        global_l = torch.empty_like(global_m)
        workspace = (partial_m, partial_l, global_m, global_l)
    else:
        if len(workspace) != 4:
            raise ValueError(f"prefill score workspace must contain four tensors, got {len(workspace)}")
        for index, (tensor, expected_shape) in enumerate(zip(workspace, expected_shapes)):
            if tensor.device != q.device or tensor.dtype != torch.float32 or tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"invalid prefill score workspace[{index}]: expected shape={expected_shape}, "
                    f"dtype=float32, device={q.device}; got shape={tuple(tensor.shape)}, "
                    f"dtype={tensor.dtype}, device={tensor.device}"
                )
        partial_m, partial_l, global_m, global_l = workspace

    common = (
        q,
        k,
        b_seq_len,
        req_to_token_indexs,
        b_req_idx,
        score_q_start,
        score_q_end,
        b_start_loc,
        b_prompt_cache_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
    )
    common_meta = {
        "H_PER_KV": kv_group_num,
        "H_KV": kv_head,
        "HEAD_BLOCKS": head_blocks,
        "candidate_start": int(candidate_start),
        "num_recent_tokens": int(num_recent_tokens),
        "sm_scale": float(head_dim) ** -0.5,
        "NUM_BLOCKS": candidate_blocks,
        "BLOCK_H": block_h,
        "BLOCK_ROWS": block_rows,
        "BLOCK_DMODEL": head_dim,
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "USE_IEEE": q.dtype == torch.float32,
        "num_warps": int(config["dot_warps"]),
        "num_stages": int(config["dot_stages"]),
    }
    if stage in {"partial", "combined"}:
        _prefill_score_partial_stats_kernel[(group_count, candidate_blocks)](
            common[0],
            common[1],
            partial_m,
            partial_l,
            *common[2:],
            **common_meta,
        )
    if stage in {"reduce", "combined"}:
        reduce_grid = (group_count, triton.cdiv(block_rows, int(config["reduce_rows"])))
        _prefill_score_reduce_stats_kernel[reduce_grid](
            partial_m,
            partial_l,
            global_m,
            global_l,
            NUM_BLOCKS=candidate_blocks,
            BLOCK_ROWS=block_rows,
            REDUCE_BLOCKS=int(config["reduce_blocks"]),
            REDUCE_ROWS=int(config["reduce_rows"]),
            num_warps=int(config["reduce_warps"]),
            num_stages=int(config["reduce_stages"]),
        )
    if stage in {"final", "combined"}:
        # Triton 3.4 does not support atomic_max on fp16/bf16 pointers. Preserve
        # cross-step max accumulation in FP32, then cast the complete result.
        atomic_score = attn_score if attn_score.dtype == torch.float32 else attn_score.float()
        _prefill_score_final_kernel[(group_count, candidate_blocks)](
            common[0],
            common[1],
            atomic_score,
            global_m,
            global_l,
            *common[2:15],
            atomic_score.stride(0),
            atomic_score.stride(1),
            *common[15:],
            **common_meta,
        )
        if atomic_score is not attn_score:
            attn_score.copy_(atomic_score)
    return workspace


@torch.no_grad()
def prefill_score_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_score: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_prompt_cache_len: torch.Tensor,
    max_query_len: int,
    req_to_token_indexs: torch.Tensor,
    score_q_start: torch.Tensor,
    score_q_end: torch.Tensor,
    *,
    candidate_start: int = 0,
    num_recent_tokens: int = 0,
    host_max_score_len: int,
    host_max_candidate_end: int,
):
    prefill_score_fwd_variant(
        q,
        k,
        attn_score,
        b_req_idx,
        b_start_loc,
        b_seq_len,
        b_prompt_cache_len,
        max_query_len,
        req_to_token_indexs,
        score_q_start,
        score_q_end,
        candidate_start=candidate_start,
        num_recent_tokens=num_recent_tokens,
        variant_id="three_pass_host_bounds_bh2",
        host_max_score_len=host_max_score_len,
        host_max_candidate_end=host_max_candidate_end,
    )
