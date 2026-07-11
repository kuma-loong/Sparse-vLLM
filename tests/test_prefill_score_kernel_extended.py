import math

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

MANDATORY_CONTEXT_LENGTHS = [
    1,
    15,
    16,
    17,
    63,
    64,
    65,
    127,
    128,
    129,
    511,
    512,
    513,
    997,
    2535,
    4093,
    4096,
    6872,
    15437,
    32749,
    65521,
    65536,
    79439,
    131071,
    262143,
]


def _score_tolerance(dtype):
    return {
        torch.float32: (5e-5, 5e-7),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (8e-3, 8e-3),
    }[dtype]


def _make_case(
    *,
    context_lens,
    windows,
    num_heads,
    num_kv_heads,
    head_dim,
    q_dtype,
    score_dtype,
    candidate_start,
    num_recent_tokens,
    slot_case="ordered",
    padded=False,
    value_case="random",
    seed=20260711,
):
    torch.manual_seed(seed)
    batch = len(context_lens)
    max_context_len = max(context_lens)
    q_starts = []
    cursor = 0
    for window in windows:
        q_starts.append(cursor)
        cursor += window
    pad = 7 if padded else 0
    q_storage = torch.empty(
        (cursor, num_heads, head_dim + pad),
        dtype=q_dtype,
        device="cuda",
    )
    q = q_storage[:, :, :head_dim]

    if slot_case == "shared":
        physical_slots = max_context_len
    elif slot_case == "gapped":
        physical_slots = sum(context_lens) * 2
    elif slot_case in {"ordered", "shuffled"}:
        physical_slots = sum(context_lens)
    else:
        raise ValueError(slot_case)
    k_storage = torch.empty(
        (physical_slots, num_kv_heads, head_dim + pad),
        dtype=q_dtype,
        device="cuda",
    )
    k_cache = k_storage[:, :, :head_dim]
    if value_case == "random":
        q.normal_()
        k_cache.normal_()
    elif value_case == "large":
        q.normal_(mean=0.0, std=20.0)
        k_cache.normal_(mean=0.0, std=20.0)
    elif value_case == "equal":
        q.fill_(1.0)
        k_cache.fill_(1.0)
    elif value_case == "dominant":
        q.zero_()
        k_cache.zero_()
        q[..., 0] = 8.0
        k_cache[0, :, 0] = 8.0
    elif value_case == "near_tie":
        q.fill_(0.25)
        k_cache.fill_(0.25)
        k_cache[0, :, 0] += 1e-3
        k_cache[1, :, 0] -= 1e-3
    else:
        raise ValueError(value_case)

    req_storage = torch.zeros(
        (batch, max_context_len + 3),
        dtype=torch.int32,
        device="cuda",
    )
    req_to_tokens = req_storage[:, 1 : max_context_len + 1]
    b_req_idx = torch.arange(batch - 1, -1, -1, dtype=torch.int32, device="cuda")
    offset = 0
    for batch_row, length in enumerate(context_lens):
        request_row = int(b_req_idx[batch_row].item())
        if slot_case == "shared":
            slots = torch.arange(length, dtype=torch.int32, device="cuda")
        elif slot_case == "gapped":
            slots = offset + torch.arange(length, dtype=torch.int32, device="cuda") * 2
            offset += length * 2
        else:
            slots = offset + torch.arange(length, dtype=torch.int32, device="cuda")
            if slot_case == "shuffled":
                slots = slots[torch.randperm(length, device="cuda")]
            offset += length
        req_to_tokens[request_row, :length] = slots

    score_storage = torch.full(
        (batch, max_context_len + 2),
        -7.0,
        dtype=score_dtype,
        device="cuda",
    )
    attn_score = score_storage[:, 1 : max_context_len + 1]
    context_tensor = torch.tensor(context_lens, dtype=torch.int32, device="cuda")
    score_ends = context_tensor.clone()
    score_starts = score_ends - torch.tensor(windows, dtype=torch.int32, device="cuda")
    return {
        "q_storage": q_storage,
        "q": q,
        "k_storage": k_storage,
        "k_cache": k_cache,
        "req_storage": req_storage,
        "req_to_tokens": req_to_tokens,
        "score_storage": score_storage,
        "attn_score": attn_score,
        "b_req_idx": b_req_idx,
        "b_start_loc": torch.tensor(q_starts, dtype=torch.int32, device="cuda"),
        "context_lens": context_tensor,
        "prompt_cache_lens": score_starts.clone(),
        "score_starts": score_starts,
        "score_ends": score_ends,
        "windows": tuple(windows),
        "context_lens_cpu": tuple(context_lens),
        "candidate_start": candidate_start,
        "num_recent_tokens": num_recent_tokens,
    }


def _run(case, *, variant_id, stage="combined", workspace=None, use_provided_bounds=False):
    from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd_variant

    return prefill_score_fwd_variant(
        case["q"],
        case["k_cache"],
        case["attn_score"],
        case["b_req_idx"],
        case["b_start_loc"],
        case["context_lens"],
        case["prompt_cache_lens"],
        max(case["windows"]),
        case["req_to_tokens"],
        case["score_starts"],
        case["score_ends"],
        candidate_start=case["candidate_start"],
        num_recent_tokens=case["num_recent_tokens"],
        variant_id=variant_id,
        stage=stage,
        workspace=workspace,
        host_max_score_len=max(case["windows"]),
        host_max_candidate_end=max(case["context_lens_cpu"]) - case["num_recent_tokens"],
        use_provided_bounds=use_provided_bounds,
    )


def _oracle(case):
    q = case["q"]
    k_cache = case["k_cache"]
    batch = len(case["context_lens_cpu"])
    num_heads = q.shape[1]
    num_kv_heads = k_cache.shape[1]
    heads_per_kv = num_heads // num_kv_heads
    out = torch.zeros(
        (batch, max(case["context_lens_cpu"])),
        dtype=torch.float32,
        device="cuda",
    )
    for batch_row, (context_len, window) in enumerate(
        zip(case["context_lens_cpu"], case["windows"])
    ):
        candidate_end = max(
            case["candidate_start"],
            context_len - case["num_recent_tokens"],
        )
        if candidate_end <= case["candidate_start"]:
            continue
        positions = torch.arange(
            case["candidate_start"],
            candidate_end,
            device="cuda",
        )
        request_row = int(case["b_req_idx"][batch_row].item())
        slots = case["req_to_tokens"][request_row, positions].long()
        q_start = int(case["b_start_loc"][batch_row].item())
        q_rows = q[q_start : q_start + window].float()
        q_positions = torch.arange(context_len - window, context_len, device="cuda")
        per_head = []
        for head in range(num_heads):
            kv_head = head // heads_per_kv
            keys = k_cache[slots, kv_head].float()
            logits = torch.matmul(q_rows[:, head], keys.T) * (q.shape[-1] ** -0.5)
            logits = logits.masked_fill(q_positions[:, None] < positions[None, :], -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            per_head.append(probabilities.sum(dim=0) / float(window))
        out[batch_row, case["candidate_start"] : candidate_end] = torch.stack(per_head).max(dim=0).values
    return out


def _assert_oracle_and_guards(case):
    torch.cuda.synchronize()
    expected = _oracle(case)
    rtol, atol = _score_tolerance(case["attn_score"].dtype)
    for row, context_len in enumerate(case["context_lens_cpu"]):
        candidate_end = max(
            case["candidate_start"],
            context_len - case["num_recent_tokens"],
        )
        actual = case["attn_score"][row]
        torch.testing.assert_close(
            actual[case["candidate_start"] : candidate_end].float(),
            expected[row, case["candidate_start"] : candidate_end],
            rtol=rtol,
            atol=atol,
        )
        assert torch.equal(actual[: case["candidate_start"]], torch.full_like(actual[: case["candidate_start"]], -7.0))
        assert torch.equal(actual[candidate_end:], torch.full_like(actual[candidate_end:], -7.0))
    assert torch.equal(case["score_storage"][:, 0], torch.full_like(case["score_storage"][:, 0], -7.0))
    assert torch.equal(case["score_storage"][:, -1], torch.full_like(case["score_storage"][:, -1], -7.0))


@pytest.mark.parametrize(
    "q_dtype,score_dtype,head_dim,num_heads,num_kv_heads,window,length,slot_case,padded",
    [
        (torch.float32, torch.float32, 16, 4, 2, 1, 17, "ordered", False),
        (torch.float16, torch.float32, 16, 4, 2, 1, 17, "ordered", False),
        (torch.bfloat16, torch.bfloat16, 32, 12, 4, 15, 63, "shuffled", False),
        (torch.float16, torch.float16, 64, 16, 4, 16, 65, "gapped", True),
        (torch.bfloat16, torch.float32, 128, 28, 4, 17, 127, "shared", True),
        (torch.bfloat16, torch.float32, 256, 32, 4, 31, 129, "ordered", False),
        (torch.float16, torch.float32, 16, 32, 2, 33, 97, "shuffled", True),
    ],
)
def test_oracle_pairwise_shape_dtype_layout_matrix(
    q_dtype,
    score_dtype,
    head_dim,
    num_heads,
    num_kv_heads,
    window,
    length,
    slot_case,
    padded,
):
    case = _make_case(
        context_lens=[length],
        windows=[window],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
        score_dtype=score_dtype,
        candidate_start=1,
        num_recent_tokens=1,
        slot_case=slot_case,
        padded=padded,
    )
    _run(case, variant_id="three_pass_current")
    _assert_oracle_and_guards(case)


@pytest.mark.parametrize("window", [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128])
def test_query_window_boundaries(window):
    case = _make_case(
        context_lens=[window + 67],
        windows=[window],
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=0,
        num_recent_tokens=0,
    )
    _run(case, variant_id="three_pass_current")
    _assert_oracle_and_guards(case)


@pytest.mark.parametrize("batch", [2, 8, 32])
def test_ragged_batch_and_request_permutation(batch):
    context_lens = [97 - (index * 7 % 29) for index in range(batch)]
    windows = [17 - (index % 3) for index in range(batch)]
    case = _make_case(
        context_lens=context_lens,
        windows=windows,
        num_heads=8,
        num_kv_heads=4,
        head_dim=16,
        q_dtype=torch.float16,
        score_dtype=torch.float32,
        candidate_start=1,
        num_recent_tokens=3,
        slot_case="gapped",
        padded=True,
    )
    _run(case, variant_id="three_pass_current")
    _assert_oracle_and_guards(case)


@pytest.mark.parametrize("value_case", ["large", "equal", "dominant", "near_tie"])
def test_numerical_stress_and_determinism(value_case):
    case = _make_case(
        context_lens=[129],
        windows=[32],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=1,
        num_recent_tokens=8,
        value_case=value_case,
    )
    outputs = []
    for _ in range(20):
        case["score_storage"].fill_(-7.0)
        _run(
            case,
            variant_id="three_pass_host_bounds",
        )
        outputs.append(case["attn_score"].clone())
    torch.cuda.synchronize()
    _assert_oracle_and_guards(case)
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])


@pytest.mark.parametrize(
    "model_shape,length,window",
    [
        ("qwen3", 4093, 32),
        ("qwen3", 32749, 128),
        ("qwen25", 4093, 32),
        ("qwen25", 79439, 128),
    ],
)
def test_host_bounds_is_bitwise_and_selection_equivalent_at_long_lengths(model_shape, length, window):
    num_heads, num_kv_heads = (32, 8) if model_shape == "qwen3" else (28, 4)
    kwargs = dict(
        context_lens=[length],
        windows=[window],
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=64,
        num_recent_tokens=512,
        slot_case="shuffled",
    )
    baseline = _make_case(**kwargs)
    candidate = _make_case(**kwargs)
    combined = _make_case(**kwargs)
    _run(baseline, variant_id="three_pass_current")
    _run(candidate, variant_id="three_pass_host_bounds")
    _run(combined, variant_id="three_pass_host_bounds_bh2")
    torch.cuda.synchronize()
    assert torch.equal(baseline["attn_score"], candidate["attn_score"])
    torch.testing.assert_close(
        combined["attn_score"],
        baseline["attn_score"],
        rtol=5e-5,
        atol=5e-7,
    )
    candidate_end = length - 512
    baseline_scores = baseline["attn_score"][0, 64:candidate_end]
    candidate_scores = candidate["attn_score"][0, 64:candidate_end]
    for requested_k in (32, 64, 128, 512, 1024):
        k = min(requested_k, baseline_scores.numel())
        assert torch.equal(
            torch.topk(baseline_scores, k, sorted=True).indices,
            torch.topk(candidate_scores, k, sorted=True).indices,
        )
        assert torch.equal(
            torch.topk(baseline_scores, k, sorted=True).indices,
            torch.topk(combined["attn_score"][0, 64:candidate_end], k, sorted=True).indices,
        )


@pytest.mark.parametrize("length", MANDATORY_CONTEXT_LENGTHS)
def test_context_length_manifest_current_and_production(length):
    window = min(15, length)
    kwargs = dict(
        context_lens=[length],
        windows=[window],
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=0,
        num_recent_tokens=0,
        slot_case="ordered",
    )
    baseline = _make_case(**kwargs)
    production = _make_case(**kwargs)

    _run(baseline, variant_id="three_pass_current")
    _run(production, variant_id="three_pass_host_bounds_bh2")
    torch.cuda.synchronize()

    assert torch.equal(baseline["attn_score"], production["attn_score"])
    assert torch.isfinite(production["attn_score"]).all()
    for requested_k in (1, 32, 128, 1024):
        k = min(requested_k, length)
        assert torch.equal(
            torch.topk(baseline["attn_score"][0], k, sorted=True).indices,
            torch.topk(production["attn_score"][0], k, sorted=True).indices,
        )
    if length <= 4096:
        _assert_oracle_and_guards(production)


def test_candidate_empty_and_single_token_ranges_preserve_guards():
    empty = _make_case(
        context_lens=[65],
        windows=[16],
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=64,
        num_recent_tokens=1,
    )
    assert _run(empty, variant_id="three_pass_current") is None
    assert torch.equal(empty["score_storage"], torch.full_like(empty["score_storage"], -7.0))

    one = _make_case(
        context_lens=[66],
        windows=[16],
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=64,
        num_recent_tokens=1,
    )
    _run(one, variant_id="three_pass_current")
    _assert_oracle_and_guards(one)


def test_host_bounds_supports_cuda_graph_with_reused_workspace():
    case = _make_case(
        context_lens=[4093],
        windows=[32],
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=64,
        num_recent_tokens=512,
    )
    workspace = _run(case, variant_id="three_pass_host_bounds")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case["attn_score"].fill_(-7.0)
        _run(
            case,
            variant_id="three_pass_host_bounds",
            workspace=workspace,
        )
    graph.replay()
    torch.cuda.synchronize()
    _assert_oracle_and_guards(case)


def test_variant_input_validation_fails_fast():
    case = _make_case(
        context_lens=[257],
        windows=[129],
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=1,
        num_recent_tokens=1,
    )
    with pytest.raises(ValueError, match="query range is too large"):
        _run(case, variant_id="three_pass_current")

    case = _make_case(
        context_lens=[97],
        windows=[17],
        num_heads=8,
        num_kv_heads=4,
        head_dim=32,
        q_dtype=torch.bfloat16,
        score_dtype=torch.float32,
        candidate_start=1,
        num_recent_tokens=1,
    )
    with pytest.raises(ValueError, match="unknown prefill score variant"):
        _run(case, variant_id="unknown")
    with pytest.raises(ValueError, match="requires both host bounds"):
        from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd_variant

        prefill_score_fwd_variant(
            case["q"],
            case["k_cache"],
            case["attn_score"],
            case["b_req_idx"],
            case["b_start_loc"],
            case["context_lens"],
            case["prompt_cache_lens"],
            max(case["windows"]),
            case["req_to_tokens"],
            case["score_starts"],
            case["score_ends"],
            candidate_start=case["candidate_start"],
            num_recent_tokens=case["num_recent_tokens"],
            variant_id="three_pass_host_bounds",
        )
