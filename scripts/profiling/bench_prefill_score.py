#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import triton

from sparsevllm.triton_kernel.prefill_score import (
    PREFILL_SCORE_VARIANTS,
    get_prefill_score_variant_config,
    prefill_score_fwd_variant,
)


VALID_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}
ARTIFACT_NAMES = (
    "run_info.json",
    "case_manifest.json",
    "raw_outputs.jsonl",
    "parsed_outputs.jsonl",
    "per_sample_results.jsonl",
    "aggregate_metrics.json",
    "compile_metadata.jsonl",
    "report.md",
)
STAGES = ("partial", "reduce", "final", "combined")
DEFAULT_SEED = 20260711
H100_BF16_PEAK_TFLOPS = 989.0
H100_HBM_TBPS = 3.35
_KERNEL_RESOURCE_CACHE = {}


def _csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def _percentile(values, percentile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _torch_dtype(name):
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]
    except KeyError as exc:
        raise ValueError(f"dtype must be fp16, bf16, or fp32, got {name!r}") from exc


def _dtype_bytes(name):
    return torch.tensor([], dtype=_torch_dtype(name)).element_size()


def _parse_head_shapes(value):
    shapes = []
    for item in _csv(value):
        parts = item.split(":")
        if len(parts) != 4:
            raise ValueError(f"head shape must be label:Hq:Hkv:D, got {item!r}")
        label, hq, hkv, head_dim = parts
        hq, hkv, head_dim = int(hq), int(hkv), int(head_dim)
        if hq <= 0 or hkv <= 0 or hq % hkv:
            raise ValueError(f"invalid head shape {item!r}")
        shapes.append((label, hq, hkv, head_dim))
    return shapes


def _ragged_values(maximum, batch, *, minimum, salt):
    if batch == 1:
        return (maximum,)
    span = max(1, min(997, maximum - minimum))
    return tuple(max(minimum, maximum - ((index * salt) % (span + 1))) for index in range(batch))


@dataclass(frozen=True)
class Case:
    case_id: str
    required: bool
    variant_id: str
    stage: str
    model_shape: str
    dtype: str
    score_dtype: str
    B: int
    Hq: int
    Hkv: int
    gqa_ratio: int
    head_dim: int
    max_context_len: int
    context_lens: tuple[int, ...]
    score_windows: tuple[int, ...]
    candidate_start: int
    num_recent_tokens: int
    slot_case: str
    layout_case: str
    seed: int
    block_m: int
    block_n: int
    block_h: int
    block_rows: int
    dot_warps: int
    dot_stages: int
    reduce_blocks: int
    reduce_rows: int
    reduce_warps: int
    reduce_stages: int
    candidate_blocks: int
    group_count: int
    stats_workspace_bytes: int
    atomic_score_bytes: int
    workspace_bytes: int
    kernel_launch_count: int
    warmup: int
    rounds: int
    iterations: int


class ArtifactWriter:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        for name in ARTIFACT_NAMES:
            path = self.run_dir / name
            path.write_text("" if name.endswith(".jsonl") or name == "report.md" else "{}\n", encoding="utf-8")

    def write_json(self, name, value):
        (self.run_dir / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def append_jsonl(self, name, value):
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")


def build_manifest(args):
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - set(PREFILL_SCORE_VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    stages = _csv(args.stages)
    if unknown_stages := sorted(set(stages) - set(STAGES)):
        raise ValueError(f"unknown stages: {unknown_stages}")

    cases = []
    for model_shape, hq, hkv, head_dim in _parse_head_shapes(args.head_shapes):
        for batch in _csv(args.batch_sizes, int):
            for max_context_len in _csv(args.context_lens, int):
                for window in _csv(args.windows, int):
                    minimum = args.candidate_start + args.num_recent_tokens + window
                    if max_context_len < minimum:
                        raise ValueError(
                            f"context length {max_context_len} is smaller than required minimum {minimum}"
                        )
                    context_lens = _ragged_values(max_context_len, batch, minimum=minimum, salt=131)
                    score_windows = _ragged_values(window, batch, minimum=1, salt=7)
                    max_candidate_end = max(length - args.num_recent_tokens for length in context_lens)
                    for dtype in _csv(args.dtypes):
                        for score_dtype in _csv(args.score_dtypes):
                            for slot_case in _csv(args.slot_cases):
                                for layout_case in _csv(args.layout_cases):
                                    for variant_id in variants:
                                        initial_blocks = triton.cdiv(
                                            max_candidate_end,
                                            64 if head_dim >= 128 else 128,
                                        )
                                        config = get_prefill_score_variant_config(
                                            variant_id,
                                            head_dim=head_dim,
                                            max_score_len=max(score_windows),
                                            kv_group_num=hq // hkv,
                                            candidate_blocks=initial_blocks,
                                        )
                                        candidate_blocks = triton.cdiv(max_candidate_end, int(config["block_n"]))
                                        config = get_prefill_score_variant_config(
                                            variant_id,
                                            head_dim=head_dim,
                                            max_score_len=max(score_windows),
                                            kv_group_num=hq // hkv,
                                            candidate_blocks=candidate_blocks,
                                        )
                                        head_blocks = triton.cdiv(hq // hkv, int(config["block_h"]))
                                        group_count = batch * hkv * head_blocks
                                        stats_workspace_bytes = (
                                            2 * group_count * candidate_blocks * int(config["block_rows"]) * 4
                                            + 2 * group_count * int(config["block_rows"]) * 4
                                        )
                                        atomic_score_bytes = (
                                            0
                                            if score_dtype == "fp32"
                                            else batch * max_context_len * 4
                                        )
                                        for stage in stages:
                                            canonical = {
                                                "variant_id": variant_id,
                                                "stage": stage,
                                                "model_shape": model_shape,
                                                "dtype": dtype,
                                                "score_dtype": score_dtype,
                                                "B": batch,
                                                "context_lens": context_lens,
                                                "score_windows": score_windows,
                                                "slot_case": slot_case,
                                                "layout_case": layout_case,
                                                "seed": args.seed,
                                            }
                                            cases.append(
                                                Case(
                                                    case_id=_json_hash(canonical)[:16],
                                                    required=True,
                                                    variant_id=variant_id,
                                                    stage=stage,
                                                    model_shape=model_shape,
                                                    dtype=dtype,
                                                    score_dtype=score_dtype,
                                                    B=batch,
                                                    Hq=hq,
                                                    Hkv=hkv,
                                                    gqa_ratio=hq // hkv,
                                                    head_dim=head_dim,
                                                    max_context_len=max_context_len,
                                                    context_lens=context_lens,
                                                    score_windows=score_windows,
                                                    candidate_start=args.candidate_start,
                                                    num_recent_tokens=args.num_recent_tokens,
                                                    slot_case=slot_case,
                                                    layout_case=layout_case,
                                                    seed=args.seed,
                                                    block_m=int(config["block_m"]),
                                                    block_n=int(config["block_n"]),
                                                    block_h=int(config["block_h"]),
                                                    block_rows=int(config["block_rows"]),
                                                    dot_warps=int(config["dot_warps"]),
                                                    dot_stages=int(config["dot_stages"]),
                                                    reduce_blocks=int(config["reduce_blocks"]),
                                                    reduce_rows=int(config["reduce_rows"]),
                                                    reduce_warps=int(config["reduce_warps"]),
                                                    reduce_stages=int(config["reduce_stages"]),
                                                    candidate_blocks=candidate_blocks,
                                                    group_count=group_count,
                                                    stats_workspace_bytes=stats_workspace_bytes,
                                                    atomic_score_bytes=atomic_score_bytes,
                                                    workspace_bytes=stats_workspace_bytes + atomic_score_bytes,
                                                    kernel_launch_count=(
                                                        (3 if stage == "combined" else 1)
                                                        + (2 if score_dtype != "fp32" and stage in {"final", "combined"} else 0)
                                                    ),
                                                    warmup=args.warmup,
                                                    rounds=args.rounds,
                                                    iterations=args.iterations,
                                                )
                                            )
    random.Random(args.seed).shuffle(cases)
    return cases


def _make_tensors(case):
    torch.manual_seed(case.seed + case.B * 1000003 + case.max_context_len * 101 + case.Hq)
    dtype = _torch_dtype(case.dtype)
    score_dtype = _torch_dtype(case.score_dtype)
    context_lens = torch.tensor(case.context_lens, dtype=torch.int32, device="cuda")
    score_ends = context_lens.clone()
    score_starts = score_ends - torch.tensor(case.score_windows, dtype=torch.int32, device="cuda")
    prompt_cache_lens = score_starts.clone()
    q_lens = tuple(case.score_windows)
    q_starts = []
    cursor = 0
    for length in q_lens:
        q_starts.append(cursor)
        cursor += length
    b_start_loc = torch.tensor(q_starts, dtype=torch.int32, device="cuda")

    padding = 13 if case.layout_case == "padded" else 0
    if case.layout_case not in {"contiguous", "padded"}:
        raise ValueError(f"layout_case must be contiguous or padded, got {case.layout_case!r}")
    q_storage = torch.randn((cursor, case.Hq, case.head_dim + padding), dtype=dtype, device="cuda")
    q = q_storage[:, :, : case.head_dim]

    if case.slot_case == "shared":
        physical_slots = case.max_context_len
    elif case.slot_case == "gapped":
        physical_slots = sum(case.context_lens) * 2
    elif case.slot_case in {"ordered", "shuffled"}:
        physical_slots = sum(case.context_lens)
    else:
        raise ValueError(f"unknown slot_case {case.slot_case!r}")
    k_storage = torch.randn(
        (physical_slots, case.Hkv, case.head_dim + padding),
        dtype=dtype,
        device="cuda",
    )
    k_cache = k_storage[:, :, : case.head_dim]
    req_to_tokens = torch.zeros((case.B, case.max_context_len), dtype=torch.int32, device="cuda")
    offset = 0
    for row, length in enumerate(case.context_lens):
        if case.slot_case == "shared":
            slots = torch.arange(length, dtype=torch.int32, device="cuda")
        elif case.slot_case == "gapped":
            slots = offset + torch.arange(length, dtype=torch.int32, device="cuda") * 2
            offset += length * 2
        else:
            slots = offset + torch.arange(length, dtype=torch.int32, device="cuda")
            if case.slot_case == "shuffled":
                permutation = torch.randperm(length, device="cuda")
                slots = slots[permutation]
            offset += length
        req_to_tokens[row, :length] = slots

    score_storage = torch.full(
        (case.B, case.max_context_len + padding),
        -7.0,
        dtype=score_dtype,
        device="cuda",
    )
    attn_score = score_storage[:, : case.max_context_len]
    return {
        "q_storage": q_storage,
        "q": q,
        "k_storage": k_storage,
        "k_cache": k_cache,
        "attn_score_storage": score_storage,
        "attn_score": attn_score,
        "req_to_tokens": req_to_tokens,
        "b_req_idx": torch.arange(case.B, dtype=torch.int32, device="cuda"),
        "b_start_loc": b_start_loc,
        "context_lens": context_lens,
        "prompt_cache_lens": prompt_cache_lens,
        "score_starts": score_starts,
        "score_ends": score_ends,
    }


def _launch(case, tensors, *, stage, workspace=None, variant_id=None, force_host_bounds=False):
    return prefill_score_fwd_variant(
        tensors["q"],
        tensors["k_cache"],
        tensors["attn_score"],
        tensors["b_req_idx"],
        tensors["b_start_loc"],
        tensors["context_lens"],
        tensors["prompt_cache_lens"],
        max(case.score_windows),
        tensors["req_to_tokens"],
        tensors["score_starts"],
        tensors["score_ends"],
        candidate_start=case.candidate_start,
        num_recent_tokens=case.num_recent_tokens,
        variant_id=variant_id or case.variant_id,
        stage=stage,
        workspace=workspace,
        host_max_score_len=max(case.score_windows),
        host_max_candidate_end=max(case.context_lens) - case.num_recent_tokens,
        use_provided_bounds=force_host_bounds,
    )


def _timed_round(function, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _cuobjdump_path():
    candidates = (
        shutil.which("cuobjdump"),
        "/usr/local/cuda-12.8/bin/cuobjdump",
        "/usr/local/cuda-12.9/bin/cuobjdump",
        "/usr/local/cuda-13.0/bin/cuobjdump",
    )
    return next((path for path in candidates if path and Path(path).is_file()), None)


def _triton_metadata_snapshot():
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir or not Path(cache_dir).is_dir():
        return {}
    return {
        str(path): path.stat().st_mtime_ns
        for path in Path(cache_dir).rglob("*.json")
    }


def _kernel_resource_key(case, kernel_name):
    return (
        kernel_name,
        case.dtype,
        case.score_dtype,
        case.Hq,
        case.Hkv,
        case.head_dim,
        case.candidate_start,
        case.num_recent_tokens,
        case.layout_case,
        case.block_m,
        case.block_n,
        case.block_h,
        case.block_rows,
        case.dot_warps,
        case.dot_stages,
        case.reduce_blocks,
        case.reduce_rows,
        case.reduce_warps,
        case.reduce_stages,
        case.candidate_blocks,
    )


def _read_kernel_resource(metadata_path, kernel_name, cuobjdump):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resource = {
        "kernel_name": kernel_name,
        "metadata_path": str(metadata_path),
        "shared_bytes": metadata.get("shared"),
        "num_warps": metadata.get("num_warps"),
        "num_stages": metadata.get("num_stages"),
        "global_scratch_size": metadata.get("global_scratch_size"),
    }
    cubin = metadata_path.with_suffix(".cubin")
    if cuobjdump and cubin.is_file():
        result = subprocess.run(
            [cuobjdump, "--dump-resource-usage", str(cubin)],
            text=True,
            capture_output=True,
            check=False,
        )
        match = re.search(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)", result.stdout)
        if result.returncode == 0 and match:
            resource.update(
                registers_per_thread=int(match.group(1)),
                stack_bytes=int(match.group(2)),
                static_shared_bytes=int(match.group(3)),
                local_bytes=int(match.group(4)),
            )
    return resource


def _compiled_resource_metadata(case, before_snapshot):
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir or not Path(cache_dir).is_dir():
        return None, "TRITON_CACHE_DIR is missing or does not exist"
    names = {
        "partial": ["_prefill_score_partial_stats_kernel"],
        "reduce": ["_prefill_score_reduce_stats_kernel"],
        "final": ["_prefill_score_final_kernel"],
        "combined": [
            "_prefill_score_partial_stats_kernel",
            "_prefill_score_reduce_stats_kernel",
            "_prefill_score_final_kernel",
        ],
    }[case.stage]
    cuobjdump = _cuobjdump_path()
    after_snapshot = _triton_metadata_snapshot()
    changed_paths = {
        path
        for path, mtime_ns in after_snapshot.items()
        if before_snapshot.get(path) != mtime_ns
    }
    resources = []
    for name in names:
        resource_key = _kernel_resource_key(case, name)
        metadata_files = [
            Path(path)
            for path in changed_paths
            if Path(path).name == f"{name}.json"
        ]
        if metadata_files:
            metadata_path = max(metadata_files, key=lambda path: path.stat().st_mtime_ns)
            resource = _read_kernel_resource(metadata_path, name, cuobjdump)
            _KERNEL_RESOURCE_CACHE[resource_key] = resource
        elif resource_key in _KERNEL_RESOURCE_CACHE:
            resource = dict(_KERNEL_RESOURCE_CACHE[resource_key])
        else:
            return None, (
                f"exact compiled metadata for {name} was not created in this launch "
                "and its compile signature was not observed earlier"
            )
        resources.append(resource)
    return resources, ""


def _selection_diagnostics(case, candidate, baseline, selection_ks):
    mismatches = []
    for row, context_len in enumerate(case.context_lens):
        candidate_end = max(case.candidate_start, context_len - case.num_recent_tokens)
        candidate_scores = candidate[row, case.candidate_start:candidate_end]
        baseline_scores = baseline[row, case.candidate_start:candidate_end]
        for requested_k in selection_ks:
            k = min(requested_k, int(candidate_scores.numel()))
            if k <= 0:
                continue
            candidate_indices = torch.topk(candidate_scores, k, sorted=True).indices
            baseline_indices = torch.topk(baseline_scores, k, sorted=True).indices
            if not torch.equal(candidate_indices, baseline_indices):
                differing = torch.nonzero(candidate_indices != baseline_indices, as_tuple=False).flatten()
                first = int(differing[0].item()) if differing.numel() else -1
                mismatches.append({"row": row, "K": k, "first_rank": first})
    return mismatches


def _performance_metrics(case, latency_ms):
    candidate_tokens = sum(
        max(0, length - case.num_recent_tokens - case.candidate_start)
        for length in case.context_lens
    )
    query_candidate_pairs = sum(
        window * max(0, length - case.num_recent_tokens - case.candidate_start)
        for length, window in zip(case.context_lens, case.score_windows)
    )
    qk_passes = 2 if case.stage == "combined" else int(case.stage in {"partial", "final"})
    flops = qk_passes * 2 * case.Hq * case.head_dim * query_candidate_pairs
    dtype_bytes = _dtype_bytes(case.dtype)
    k_bytes = qk_passes * candidate_tokens * case.Hkv * case.head_dim * dtype_bytes
    workspace_bytes = {
        "partial": case.stats_workspace_bytes // 2,
        "reduce": case.stats_workspace_bytes,
        "final": case.stats_workspace_bytes // 2 + 2 * case.atomic_score_bytes,
        "combined": case.stats_workspace_bytes * 2 + 2 * case.atomic_score_bytes,
    }[case.stage]
    seconds = latency_ms / 1000.0
    return {
        "estimated_qk_flops": flops,
        "estimated_bytes": k_bytes + workspace_bytes,
        "estimated_tflops": flops / seconds / 1e12,
        "estimated_gbps": (k_bytes + workspace_bytes) / seconds / 1e9,
        "bf16_peak_efficiency_pct": flops / seconds / 1e12 / H100_BF16_PEAK_TFLOPS * 100,
        "hbm_peak_efficiency_pct": (k_bytes + workspace_bytes) / seconds / 1e12 / H100_HBM_TBPS * 100,
    }


def run_case(case, writer, *, selection_ks, profile_only=False):
    base = asdict(case)
    compile_row = {
        **base,
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "resource_metadata_status": "unavailable",
        "resource_metadata_reason": "",
    }
    try:
        tensors = _make_tensors(case)
        base["tensor_strides"] = {
            name: list(tensor.stride())
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor)
        }
        compile_row["tensor_strides"] = base["tensor_strides"]
        torch.cuda.synchronize()
        metadata_before = _triton_metadata_snapshot()
        compile_start = time.perf_counter()
        workspace = _launch(case, tensors, stage="combined")
        torch.cuda.synchronize()
        compile_row["first_call_ms"] = (time.perf_counter() - compile_start) * 1000
        resources, resource_reason = _compiled_resource_metadata(case, metadata_before)
        if resources is None:
            compile_row["resource_metadata_reason"] = resource_reason
        else:
            compile_row["resource_metadata_status"] = "available"
            compile_row["kernels"] = resources
        compile_row["status"] = "success"
        writer.append_jsonl("compile_metadata.jsonl", compile_row)

        candidate_output = tensors["attn_score"].clone()
        if case.stage == "combined":
            baseline_tensors = dict(tensors)
            baseline_tensors["attn_score_storage"] = torch.full_like(tensors["attn_score_storage"], -7.0)
            baseline_tensors["attn_score"] = baseline_tensors["attn_score_storage"][:, : case.max_context_len]
            _launch(case, baseline_tensors, stage="combined", variant_id="three_pass_current")
            torch.cuda.synchronize()
            baseline_output = baseline_tensors["attn_score"].clone()
            max_abs_diff = float((candidate_output.float() - baseline_output.float()).abs().max().item())
            mismatches = _selection_diagnostics(case, candidate_output, baseline_output, selection_ks)
        else:
            max_abs_diff = None
            mismatches = []

        if case.stage == "combined":
            function = lambda: _launch(case, tensors, stage="combined")
        else:
            function = lambda: _launch(
                case,
                tensors,
                stage=case.stage,
                workspace=workspace,
                force_host_bounds=True,
            )
        for _ in range(case.warmup):
            function()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        function()
        torch.cuda.synchronize()
        peak_extra_bytes = max(0, torch.cuda.max_memory_allocated() - allocated_before)
        if profile_only:
            rounds_ms = []
            status = "success"
            reason = "profile-only launch completed"
        else:
            rounds_ms = [_timed_round(function, case.iterations) for _ in range(case.rounds)]
            mean_ms = statistics.fmean(rounds_ms)
            cv = statistics.pstdev(rounds_ms) / mean_ms if len(rounds_ms) > 1 else 0.0
            status = "success" if cv <= 0.03 else "metric_failed"
            reason = "" if status == "success" else f"round coefficient of variation {cv:.4f} exceeds 0.03"

        finite = all(bool(torch.isfinite(tensor).all().item()) for tensor in workspace)
        if case.stage == "combined":
            for row, context_len in enumerate(case.context_lens):
                candidate_end = max(case.candidate_start, context_len - case.num_recent_tokens)
                finite = finite and bool(
                    torch.isfinite(tensors["attn_score"][row, case.candidate_start:candidate_end]).all().item()
                )
        if not finite:
            status, reason = "metric_failed", "non-finite kernel output"
        if mismatches:
            status, reason = "metric_failed", f"selection mismatch in {len(mismatches)} row/K comparisons"

        raw = {
            **base,
            "rounds_ms": rounds_ms,
            "finite": finite,
            "peak_extra_bytes": peak_extra_bytes,
            "max_abs_diff_vs_current": max_abs_diff,
            "selection_mismatches": mismatches,
            "status": status,
            "reason": reason,
        }
        writer.append_jsonl("raw_outputs.jsonl", raw)
        parsed = {
            **base,
            "finite": finite,
            "peak_extra_bytes": peak_extra_bytes,
            "max_abs_diff_vs_current": max_abs_diff,
            "selection_mismatch_count": len(mismatches),
            "status": status,
            "reason": reason,
        }
        if rounds_ms:
            parsed.update(
                latency_p50_ms=statistics.median(rounds_ms),
                latency_p90_ms=_percentile(rounds_ms, 0.9),
                latency_min_ms=min(rounds_ms),
                latency_max_ms=max(rounds_ms),
                latency_mean_ms=statistics.fmean(rounds_ms),
                round_cv=statistics.pstdev(rounds_ms) / statistics.fmean(rounds_ms),
            )
            parsed.update(_performance_metrics(case, parsed["latency_p50_ms"]))
        writer.append_jsonl("parsed_outputs.jsonl", parsed)
        writer.append_jsonl(
            "per_sample_results.jsonl",
            {**base, "status": status, "failure_kind": None, "reason": reason},
        )
        return parsed
    except (ValueError, TypeError, AssertionError) as exc:
        status, failure_kind = "invalid_input", "validation"
        caught_exception = exc
        caught_traceback = traceback.format_exc()
    except torch.cuda.OutOfMemoryError as exc:
        status, failure_kind = "model_failed", "oom"
        caught_exception = exc
        caught_traceback = traceback.format_exc()
    except Exception as exc:
        status, failure_kind = "model_failed", "compile_or_launch"
        caught_exception = exc
        caught_traceback = traceback.format_exc()

    error = {
        **base,
        "status": status,
        "failure_kind": failure_kind,
        "reason": str(caught_exception),
        "exception_type": type(caught_exception).__name__,
        "exception_text": caught_traceback,
    }
    compile_row.update(error)
    writer.append_jsonl("compile_metadata.jsonl", compile_row)
    writer.append_jsonl("raw_outputs.jsonl", error)
    writer.append_jsonl("parsed_outputs.jsonl", error)
    writer.append_jsonl("per_sample_results.jsonl", error)
    return error


def _comparison_key(row):
    ignored = {
        "case_id",
        "variant_id",
        "block_n",
        "block_h",
        "block_rows",
        "dot_warps",
        "dot_stages",
        "reduce_blocks",
        "reduce_rows",
        "reduce_warps",
        "reduce_stages",
        "candidate_blocks",
        "group_count",
        "stats_workspace_bytes",
        "atomic_score_bytes",
        "workspace_bytes",
        "status",
        "reason",
        "tensor_strides",
        "finite",
        "peak_extra_bytes",
        "max_abs_diff_vs_current",
        "selection_mismatch_count",
        "round_cv",
        "estimated_qk_flops",
        "estimated_bytes",
        "estimated_tflops",
        "estimated_gbps",
        "bf16_peak_efficiency_pct",
        "hbm_peak_efficiency_pct",
    }
    return tuple(
        sorted(
            (key, json.dumps(value, sort_keys=True))
            for key, value in row.items()
            if key not in ignored and not key.startswith("latency_")
        )
    )


def aggregate_results(results):
    status_counts = {
        status: sum(row.get("status") == status for row in results)
        for status in sorted(VALID_STATUSES)
    }
    by_key = {}
    for row in results:
        by_key.setdefault(_comparison_key(row), {})[row["variant_id"]] = row
    comparisons = []
    for variants in by_key.values():
        baseline = variants.get("three_pass_current")
        if not baseline or baseline.get("status") != "success" or "latency_p50_ms" not in baseline:
            continue
        for variant_id, row in variants.items():
            if row.get("status") != "success" or "latency_p50_ms" not in row:
                continue
            comparisons.append(
                {
                    "case_id": row["case_id"],
                    "stage": row["stage"],
                    "model_shape": row["model_shape"],
                    "variant_id": variant_id,
                    "baseline_id": "three_pass_current",
                    "latency_ratio": row["latency_p50_ms"] / baseline["latency_p50_ms"],
                    "speedup": baseline["latency_p50_ms"] / row["latency_p50_ms"],
                }
            )
    summaries = {}
    for comparison in comparisons:
        key = (comparison["stage"], comparison["model_shape"], comparison["variant_id"])
        summaries.setdefault(key, []).append(comparison["latency_ratio"])
    performance_gates = []
    for (stage, model_shape, variant_id), ratios in summaries.items():
        geomean_ratio = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
        max_ratio = max(ratios)
        geomean_limit = 0.97 if stage == "combined" else 1.0
        max_limit = 1.05
        performance_gates.append(
            {
                "stage": stage,
                "model_shape": model_shape,
                "variant_id": variant_id,
                "baseline_id": "three_pass_current",
                "case_count": len(ratios),
                "geomean_latency_ratio": geomean_ratio,
                "geomean_speedup": 1.0 / geomean_ratio,
                "max_latency_ratio": max_ratio,
                "geomean_limit": geomean_limit,
                "max_case_limit": max_limit,
                "passed": geomean_ratio <= geomean_limit and max_ratio <= max_limit,
            }
        )
    return {
        "status_counts": status_counts,
        "required_case_count": len(results),
        "all_required_success": all(row.get("status") == "success" for row in results),
        "comparisons": comparisons,
        "performance_gates": performance_gates,
    }


def _run_info(args, manifest_hash):
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    gpu_runtime = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip().splitlines()
    return {
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "seed": args.seed,
        "case_manifest_hash": manifest_hash,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "iterations": args.iterations,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": [properties.major, properties.minor],
        "gpu_total_memory": properties.total_memory,
        "gpu_runtime_metadata": gpu_runtime,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "nsight_compute": "not used: normal-user performance counter access is unavailable",
        "selection_ks": _csv(args.selection_ks, int),
        "thresholds": {
            "fp32": {"rtol": 5e-5, "atol": 5e-7},
            "fp16": {"rtol": 2e-3, "atol": 2e-3},
            "bf16": {"rtol": 8e-3, "atol": 8e-3},
            "round_cv_max": 0.03,
        },
        "theoretical_assumptions": {
            "h100_bf16_tflops": H100_BF16_PEAK_TFLOPS,
            "h100_hbm_tbps": H100_HBM_TBPS,
        },
    }


def _report(run_info, aggregate):
    lines = [
        "# Prefill score 算子运行报告",
        "",
        "## 实验环境",
        "",
        f"- GPU：{run_info['gpu_name']}，compute capability {run_info['compute_capability']}。",
        f"- Torch / Triton / CUDA：{run_info['torch']} / {run_info['triton']} / {run_info['cuda_runtime']}。",
        f"- Git：`{run_info['git_branch']}` @ `{run_info['git_commit']}`。",
        f"- seed：{run_info['seed']}；manifest hash：`{run_info['case_manifest_hash']}`。",
        "- 按任务要求未使用 Nsight Compute；资源信息来自 Triton cache 与 cuobjdump。",
        "",
        "## 状态",
        "",
        f"required case 全部成功：{aggregate['all_required_success']}。状态统计：`{aggregate['status_counts']}`。",
        "",
        "## 性能 gate",
        "",
        "| stage | 模型形状 | candidate | case | 几何平均加速 | 最差延迟比 | 通过 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for gate in aggregate["performance_gates"]:
        lines.append(
            f"| {gate['stage']} | {gate['model_shape']} | {gate['variant_id']} | {gate['case_count']} | "
            f"{gate['geomean_speedup']:.3f}x | {gate['max_latency_ratio']:.3f} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "`partial/reduce/final` 使用预分配 workspace 和 host bounds，仅测对应 kernel；"
            "`combined` 保留各 variant 的真实 wrapper、workspace 分配和同步语义。",
            "Qwen3-8B（Hq32/Hkv8/D128）是优先性能形状，Qwen2.5-7B（Hq28/Hkv4/D128）用于泛化回归。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Reproducible prefill score Triton benchmark")
    parser.add_argument("--variants", default="three_pass_current,three_pass_host_bounds")
    parser.add_argument("--stages", default="combined")
    parser.add_argument("--head-shapes", default="qwen3_8b:32:8:128,qwen25_7b:28:4:128")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--context-lens", default="4093,32749,79439")
    parser.add_argument("--windows", default="32,128")
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--score-dtypes", default="fp32")
    parser.add_argument("--candidate-start", type=int, default=64)
    parser.add_argument("--num-recent-tokens", type=int, default=512)
    parser.add_argument("--slot-cases", default="ordered,shuffled")
    parser.add_argument("--layout-cases", default="contiguous")
    parser.add_argument("--selection-ks", default="32,64,128,512,1024")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for prefill score benchmarking")
    if not args.profile_only and (args.warmup < 25 or args.rounds < 5 or args.iterations < 100):
        raise ValueError("formal timing requires warmup>=25, rounds>=5, and iterations>=100")
    manifest = build_manifest(args)
    manifest_rows = [asdict(case) for case in manifest]
    manifest_hash = _json_hash(manifest_rows)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + manifest_hash[:8]
    run_dir = args.output_dir / "prefill_score" / run_id
    writer = ArtifactWriter(run_dir)
    writer.write_json("case_manifest.json", {"sha256": manifest_hash, "cases": manifest_rows})
    run_info = _run_info(args, manifest_hash)
    run_info["run_id"] = run_id
    writer.write_json("run_info.json", run_info)
    results = []
    selection_ks = _csv(args.selection_ks, int)
    for index, case in enumerate(manifest, start=1):
        print(
            f"[{index}/{len(manifest)}] {case.case_id} {case.model_shape} {case.stage} {case.variant_id}",
            flush=True,
        )
        results.append(
            run_case(case, writer, selection_ks=selection_ks, profile_only=args.profile_only)
        )
        torch.cuda.empty_cache()
    aggregate = aggregate_results(results)
    writer.write_json("aggregate_metrics.json", aggregate)
    (run_dir / "report.md").write_text(_report(run_info, aggregate), encoding="utf-8")
    print(f"artifacts={run_dir}")
    if not aggregate["all_required_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
