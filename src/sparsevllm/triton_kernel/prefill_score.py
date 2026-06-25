import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_score_kernel(
    Q,
    K,
    Attn_Score,
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
    kv_group_num: tl.constexpr,
    candidate_start: tl.constexpr,
    num_recent_tokens: tl.constexpr,
    sm_scale: tl.constexpr,
    H: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    cur_bh = tl.program_id(0)
    cur_batch = cur_bh // H
    cur_head = cur_bh % H
    cur_kv_head = cur_head // kv_group_num

    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    prompt_cache_len = tl.load(B_Prompt_Cache_Len + cur_batch)
    context_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch) - prompt_cache_len
    cur_batch_req_idx = tl.load(B_req_idx + cur_batch)
    score_q_start = tl.load(Score_Q_Start + cur_batch)
    score_q_end = tl.load(Score_Q_End + cur_batch)
    score_q_len = tl.maximum(score_q_end - score_q_start, 1)

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    q_abs_pos = score_q_start + offs_m
    q_rel_pos = q_abs_pos - prompt_cache_len
    q_in_score_range = (
        (q_abs_pos < score_q_end)
        & (q_rel_pos >= 0)
        & (q_rel_pos < cur_batch_seq_len)
    )

    off_q = (
        (cur_batch_in_all_start_index + q_rel_pos[:, None]) * stride_qt
        + cur_head * stride_qh
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(Q + off_q, mask=q_in_score_range[:, None], other=0.0)

    candidate_end = tl.maximum(candidate_start, context_len - num_recent_tokens)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1.0e20
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # First pass: online softmax statistics over candidate KV tokens.
    for start_n in range(0, candidate_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_pos = start_n + offs_n
        kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
            mask=kv_in_candidate,
            other=0,
        )
        off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale
        causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
        valid = q_in_score_range[:, None] & kv_in_candidate[None, :] & causal_mask
        qk = tl.where(valid, qk, -1.0e20)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    # Second pass: sum softmax probabilities over the selected query window,
    # divide by query count, then max-reduce across heads into [B, L].
    safe_l_i = tl.where(l_i > 0.0, l_i, 1.0)
    for start_n in range(0, candidate_end, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        kv_pos = start_n + offs_n
        kv_in_candidate = (kv_pos >= candidate_start) & (kv_pos < candidate_end)
        kv_loc = tl.load(
            Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx + stride_req_to_tokens_s * kv_pos,
            mask=kv_in_candidate,
            other=0,
        )
        off_k = kv_loc[None, :] * stride_ks + cur_kv_head * stride_kh + offs_d[:, None] * stride_kd
        k = tl.load(K + off_k, mask=kv_in_candidate[None, :], other=0.0)
        qk = tl.dot(q, k) * sm_scale
        causal_mask = q_abs_pos[:, None] >= kv_pos[None, :]
        valid = q_in_score_range[:, None] & kv_in_candidate[None, :] & causal_mask
        probs = tl.exp(qk - m_i[:, None]) / safe_l_i[:, None]
        probs = tl.where(valid, probs, 0.0)
        token_score = tl.sum(probs, axis=0) / (score_q_len * 1.0)
        tl.atomic_max(
            Attn_Score + cur_batch * stride_asb + kv_pos * stride_asl,
            token_score,
            mask=kv_in_candidate,
        )


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
):
    head_dim = q.shape[-1]
    assert k.shape[-1] == head_dim
    assert q.dtype == k.dtype
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    assert attn_score.dim() == 2
    assert head_dim in {16, 32, 64, 128, 256}
    batch, head = b_seq_len.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k.shape[1]
    if kv_group_num <= 0 or q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"num query heads must be divisible by num kv heads: q={q.shape[1]} k={k.shape[1]}")
    max_score_len = int((score_q_end - score_q_start).max().item())
    if max_score_len <= 0:
        return
    block_m = max(16, triton.next_power_of_2(max_score_len))
    if block_m > 128:
        raise ValueError(f"prefill score query range is too large for this kernel: {max_score_len} > 128")
    block_n = 128
    grid = (batch * head, 1, 1)
    if grid[0] <= 0:
        return
    _prefill_score_kernel[grid](
        q,
        k,
        attn_score,
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
        attn_score.stride(0),
        attn_score.stride(1),
        req_to_token_indexs.stride(0),
        req_to_token_indexs.stride(1),
        kv_group_num=kv_group_num,
        candidate_start=int(candidate_start),
        num_recent_tokens=int(num_recent_tokens),
        sm_scale=float(head_dim) ** -0.5,
        H=head,
        BLOCK_DMODEL=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if head_dim <= 64 else 8,
        num_stages=1,
    )
