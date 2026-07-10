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

from sparsevllm.triton_kernel.flash_decoding_stage2 import (
    flash_decode_stage2,
    flash_decode_stage2_variant,
)
from sparsevllm.triton_kernel.gqa_flash_decoding_stage1 import (
    flash_decode_stage1_variant,
    get_stage1_variant_config,
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
STAGE1_VARIANTS = (
    "grouped_s1_allow256_w2",
    "grouped_s1_allow256_w4",
    "grouped_s1_allow256_w8",
    "grouped_s1_bn16_w2_s1",
    "grouped_s1_bn16_w2_s2",
    "grouped_s1_bn16_w2_s3",
    "grouped_s1_bn32_w2_s2",
    "grouped_s1_bn64_w2_s2",
    "grouped_s1_bn128_w2_s2",
    "per_q_s1_w4",
    "per_q_s1_w8",
)
STAGE2_VARIANTS = ("unified_s2_w4", "unified_s2_w8")
H100_BF16_PEAK_TFLOPS = 989.0
H100_HBM_TBPS = 3.35
DEFAULT_SEED = 20260710
STANDARD_SEQ_LENS = (256, 1024, 8192, 32768, 131072, 262144)
BROAD_SEQ_LENS = (
    15,
    16,
    17,
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
)


def _csv(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _json_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_seq_lens(args):
    if args.seq_lens:
        return _csv(args.seq_lens, int)
    profile = getattr(args, "seq_profile", "standard")
    if profile == "standard":
        return list(STANDARD_SEQ_LENS)
    if profile == "broad":
        return list(BROAD_SEQ_LENS)
    raise ValueError(f"Unknown seq_profile {profile!r}.")


def _git(*args):
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"


def _command_version(command):
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = (result.stdout or result.stderr).strip()
    return output if result.returncode == 0 else f"unavailable: {output}"


def _ncu_path():
    candidates = [
        os.environ.get("NCU"),
        "/opt/nvidia/nsight-compute/2026.1.0/ncu",
        "ncu",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        result = subprocess.run([candidate, "--version"], text=True, capture_output=True, check=False)
        if result.returncode == 0:
            return candidate, result.stdout.strip()
    return None, "unavailable: ncu executable not found"


def _cuobjdump_path():
    candidates = [
        shutil.which("cuobjdump"),
        "/usr/local/cuda-12.9/bin/cuobjdump",
        "/usr/local/cuda-13.0/bin/cuobjdump",
        "/usr/local/cuda-13.2/bin/cuobjdump",
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)


def _compiled_resource_metadata(case):
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if not cache_dir or not Path(cache_dir).is_dir():
        return None, "TRITON_CACHE_DIR is missing or does not exist"
    names = []
    if case.stage in {"stage1", "combined"}:
        schedule = case.variant_id.split("+", 1)[0]
        suffix = "grouped" if schedule.startswith("grouped") else "per_q"
        names.append(f"_fwd_kernel_gqa_flash_decode_stage1_{suffix}")
    if case.stage in {"stage2", "combined"}:
        names.append("_fwd_kernel_flash_decode_stage2")
    cuobjdump = _cuobjdump_path()
    resources = []
    for name in names:
        metadata_files = list(Path(cache_dir).rglob(f"{name}.json"))
        if not metadata_files:
            return None, f"compiled metadata for {name} was not found in {cache_dir}"
        metadata_path = max(metadata_files, key=lambda path: path.stat().st_mtime_ns)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        resource = {
            "kernel_name": name,
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
                    {
                        "registers_per_thread": int(match.group(1)),
                        "stack_bytes": int(match.group(2)),
                        "static_shared_bytes": int(match.group(3)),
                        "local_bytes": int(match.group(4)),
                    }
                )
        resources.append(resource)
    return resources, ""


@dataclass(frozen=True)
class Case:
    case_id: str
    required: bool
    variant_id: str
    stage: str
    score_mode: str
    dtype: str
    score_dtype: str
    B: int
    Hq: int
    Hkv: int
    gqa_ratio: int
    head_dim: int
    context_len: int
    max_len_in_batch: int
    block_seq: int
    block_n: int
    slot_case: str
    layout_case: str
    seed: int
    num_warps: int
    num_stages: int
    warmup: int
    rounds: int
    iterations: int


class ArtifactWriter:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "ncu").mkdir()
        (self.run_dir / "nsys").mkdir()
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


def _variant_config(variant):
    if variant in STAGE1_VARIANTS:
        return get_stage1_variant_config(variant)
    return {
        "num_warps": int(variant.rsplit("w", 1)[1]),
        "block_n": 16,
        "num_stages": 2,
    }


def build_manifest(args):
    variants = _csv(args.variants)
    unknown = sorted(set(variants) - set(STAGE1_VARIANTS) - set(STAGE2_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    stages = _csv(args.stages)
    dtypes = _csv(args.dtypes)
    score_modes = _csv(args.score_modes)
    cases = []
    for batch in _csv(args.batch_sizes, int):
        for seq_len in _resolve_seq_lens(args):
            for head_dim in _csv(args.head_dims, int):
                for dtype in dtypes:
                    for block_seq in _csv(args.block_seqs, int):
                        for slot_case in _csv(args.slot_orders):
                            for score_mode in score_modes:
                                for stage in stages:
                                    if stage == "stage1":
                                        stage_variants = [v for v in variants if v in STAGE1_VARIANTS]
                                    elif stage == "stage2":
                                        stage_variants = [v for v in variants if v in STAGE2_VARIANTS]
                                    elif stage == "combined":
                                        suffix = "unified_s2_w8" if _next_power_of_two(head_dim) >= 256 else "unified_s2_w4"
                                        stage_variants = [f"{v}+{suffix}" for v in variants if v in STAGE1_VARIANTS]
                                    else:
                                        raise ValueError(f"Unknown stage {stage!r}")
                                    for variant in stage_variants:
                                        stage1_or_stage2_variant = variant.split("+", 1)[0]
                                        config = _variant_config(stage1_or_stage2_variant)
                                        canonical = {
                                            "variant_id": variant,
                                            "stage": stage,
                                            "score_mode": score_mode,
                                            "dtype": dtype,
                                            "B": batch,
                                            "Hq": args.num_heads,
                                            "Hkv": args.num_kv_heads,
                                            "head_dim": head_dim,
                                            "context_len": seq_len,
                                            "block_seq": block_seq,
                                            "slot_case": slot_case,
                                            "seed": args.seed,
                                        }
                                        case_id = _json_hash(canonical)[:16]
                                        cases.append(
                                            Case(
                                                case_id=case_id,
                                                required=True,
                                                variant_id=variant,
                                                stage=stage,
                                                score_mode=score_mode,
                                                dtype=dtype,
                                                score_dtype="fp32",
                                                B=batch,
                                                Hq=args.num_heads,
                                                Hkv=args.num_kv_heads,
                                                gqa_ratio=args.num_heads // args.num_kv_heads,
                                                head_dim=head_dim,
                                                context_len=seq_len,
                                                max_len_in_batch=seq_len,
                                                block_seq=block_seq,
                                                block_n=config["block_n"],
                                                slot_case=slot_case,
                                                layout_case="contiguous",
                                                seed=args.seed,
                                                num_warps=config["num_warps"],
                                                num_stages=config["num_stages"],
                                                warmup=args.warmup,
                                                rounds=args.rounds,
                                                iterations=args.iterations,
                                            )
                                        )
    random.Random(args.seed).shuffle(cases)
    return cases


def _torch_dtype(name):
    try:
        return {"fp16": torch.float16, "bf16": torch.bfloat16}[name]
    except KeyError as exc:
        raise ValueError(f"dtype must be fp16 or bf16, got {name!r}") from exc


def _make_tensors(case):
    torch.manual_seed(case.seed + case.B * 1000003 + case.context_len * 101 + case.head_dim)
    dtype = _torch_dtype(case.dtype)
    blocks = math.ceil(case.context_len / case.block_seq)
    context_lens = torch.full((case.B,), case.context_len, dtype=torch.int32, device="cuda")
    if case.stage == "stage2":
        return {
            "mid_o": torch.randn(
                (case.B, case.Hq, blocks, case.head_dim), dtype=torch.float32, device="cuda"
            ),
            "mid_lse": torch.randn((case.B, case.Hq, blocks), dtype=torch.float32, device="cuda"),
            "context_lens": context_lens,
            "output": torch.empty((case.B, case.Hq, case.head_dim), dtype=dtype, device="cuda"),
        }
    q = torch.randn((case.B, case.Hq, case.head_dim), dtype=dtype, device="cuda")
    slots_total = case.B * case.context_len
    k = torch.randn((slots_total, case.Hkv, case.head_dim), dtype=dtype, device="cuda")
    v = torch.randn_like(k)
    req_to_tokens = torch.arange(slots_total, dtype=torch.int32, device="cuda").view(case.B, case.context_len)
    if case.slot_case == "shuffled":
        req_to_tokens = torch.stack(
            [
                torch.randperm(case.context_len, device="cuda", dtype=torch.int64).to(torch.int32)
                + batch_idx * case.context_len
                for batch_idx in range(case.B)
            ]
        )
    elif case.slot_case != "ordered":
        raise ValueError(f"slot_case must be ordered or shuffled, got {case.slot_case!r}")
    req_indices = torch.arange(case.B, dtype=torch.int32, device="cuda")
    mid_o = torch.empty((case.B, case.Hq, blocks, case.head_dim), dtype=torch.float32, device="cuda")
    mid_lse = torch.empty((case.B, case.Hq, blocks), dtype=torch.float32, device="cuda")
    output = torch.empty((case.B, case.Hq, case.head_dim), dtype=dtype, device="cuda")
    if case.score_mode == "3d":
        score = torch.full((case.B, case.Hq, case.context_len), -float("inf"), dtype=torch.float32, device="cuda")
    elif case.score_mode == "2d":
        score = torch.full((case.B, case.context_len), -float("inf"), dtype=torch.float32, device="cuda")
    elif case.score_mode == "none":
        score = None
    else:
        raise ValueError(f"score_mode must be none, 2d, or 3d, got {case.score_mode!r}")
    return {
        "q": q,
        "k": k,
        "v": v,
        "req_to_tokens": req_to_tokens,
        "req_indices": req_indices,
        "context_lens": context_lens,
        "mid_o": mid_o,
        "mid_lse": mid_lse,
        "output": output,
        "score": score,
    }


def _make_callable(case, tensors):
    stage1_variant = case.variant_id.split("+", 1)[0]
    stage2_variant = case.variant_id.split("+", 1)[1] if "+" in case.variant_id else case.variant_id

    def stage1():
        flash_decode_stage1_variant(
            tensors["q"],
            tensors["k"],
            tensors["v"],
            tensors["req_to_tokens"],
            tensors["req_indices"],
            tensors["context_lens"],
            case.max_len_in_batch,
            tensors["mid_o"],
            tensors["mid_lse"],
            case.block_seq,
            variant_id=stage1_variant,
            attn_score=tensors["score"],
        )

    def stage2():
        flash_decode_stage2_variant(
            tensors["mid_o"],
            tensors["mid_lse"],
            tensors["context_lens"],
            tensors["output"],
            case.block_seq,
            variant_id=stage2_variant,
        )

    if case.stage == "stage1":
        return stage1
    if case.stage == "stage2":
        return stage2
    if case.stage == "combined":
        def combined():
            stage1()
            stage2()
        return combined
    raise ValueError(f"Unknown stage {case.stage!r}")


def _percentile(values, percentile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _timed_round(function, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _performance_metrics(case, latency_ms):
    if case.stage == "stage2":
        flops = case.B * case.Hq * math.ceil(case.context_len / case.block_seq) * case.head_dim * 3
        bytes_moved = case.B * case.Hq * math.ceil(case.context_len / case.block_seq) * (case.head_dim + 1) * 4
    else:
        flops = 4 * case.B * case.Hq * case.context_len * case.head_dim
        bytes_moved = (
            case.B * case.Hq * case.head_dim * 2
            + 2 * case.B * case.context_len * case.Hkv * case.head_dim * 2
        )
    seconds = latency_ms / 1000
    return {
        "estimated_tflops": flops / seconds / 1e12,
        "estimated_gbps": bytes_moved / seconds / 1e9,
        "bf16_peak_efficiency_pct": (flops / seconds / 1e12) / H100_BF16_PEAK_TFLOPS * 100,
        "hbm_peak_efficiency_pct": (bytes_moved / seconds / 1e12) / H100_HBM_TBPS * 100,
    }


def run_case(case, writer, profile_only=False):
    base = asdict(case)
    base["context_lens"] = [case.context_len] * case.B
    compile_row = {
        **base,
        "resource_metadata_status": "unavailable",
        "resource_metadata_reason": "Triton 3.4 launcher does not expose stable register/spill metadata; use NCU reports.",
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    try:
        tensors = _make_tensors(case)
        base["tensor_strides"] = {
            name: list(tensor.stride())
            for name, tensor in tensors.items()
            if isinstance(tensor, torch.Tensor)
        }
        compile_row["tensor_strides"] = base["tensor_strides"]
        function = _make_callable(case, tensors)
        torch.cuda.synchronize()
        compile_start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        compile_row["first_call_ms"] = (time.perf_counter() - compile_start) * 1000
        compile_row["status"] = "success"
        resources, resource_reason = _compiled_resource_metadata(case)
        if resources is not None:
            compile_row["resource_metadata_status"] = "available"
            compile_row["resource_metadata_reason"] = ""
            compile_row["kernels"] = resources
        else:
            compile_row["resource_metadata_reason"] = resource_reason
        writer.append_jsonl("compile_metadata.jsonl", compile_row)
        for _ in range(case.warmup):
            function()
        torch.cuda.synchronize()
        if profile_only:
            function()
            torch.cuda.synchronize()
            rounds_ms = []
            status = "success"
            reason = "profile-only launch completed"
        else:
            rounds_ms = [_timed_round(function, case.iterations) for _ in range(case.rounds)]
            mean_ms = statistics.fmean(rounds_ms)
            cv = statistics.pstdev(rounds_ms) / mean_ms if len(rounds_ms) > 1 else 0.0
            status = "success" if cv <= 0.03 else "metric_failed"
            reason = "" if status == "success" else f"round coefficient of variation {cv:.4f} exceeds 0.03"
        finite = bool(torch.isfinite(tensors["mid_o"]).all().item() and torch.isfinite(tensors["mid_lse"]).all().item())
        if case.stage in {"stage2", "combined"}:
            finite = finite and bool(torch.isfinite(tensors["output"]).all().item())
        if not finite:
            status = "metric_failed"
            reason = "non-finite kernel output"
        raw = {**base, "rounds_ms": rounds_ms, "finite": finite, "status": status, "reason": reason}
        writer.append_jsonl("raw_outputs.jsonl", raw)
        parsed = {**base, "status": status, "reason": reason, "finite": finite}
        if rounds_ms:
            parsed.update(
                {
                    "latency_p50_ms": statistics.median(rounds_ms),
                    "latency_p90_ms": _percentile(rounds_ms, 0.9),
                    "latency_min_ms": min(rounds_ms),
                    "latency_max_ms": max(rounds_ms),
                    "latency_mean_ms": statistics.fmean(rounds_ms),
                    "round_cv": statistics.pstdev(rounds_ms) / statistics.fmean(rounds_ms),
                }
            )
            parsed.update(_performance_metrics(case, parsed["latency_p50_ms"]))
        writer.append_jsonl("parsed_outputs.jsonl", parsed)
        writer.append_jsonl("per_sample_results.jsonl", {**base, "status": status, "failure_kind": None, "reason": reason})
        return parsed
    except (ValueError, TypeError) as exc:
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
    ignored = {"case_id", "variant_id", "num_warps", "status", "reason"}
    return tuple(sorted((key, json.dumps(value, sort_keys=True)) for key, value in row.items() if key not in ignored and not key.startswith("latency_") and key not in {"round_cv", "finite", "estimated_tflops", "estimated_gbps", "bf16_peak_efficiency_pct", "hbm_peak_efficiency_pct"}))


def aggregate_results(results):
    status_counts = {status: sum(row.get("status") == status for row in results) for status in sorted(VALID_STATUSES)}
    by_key = {}
    for row in results:
        by_key.setdefault(_comparison_key(row), {})[row["variant_id"]] = row
    comparisons = []
    for variants in by_key.values():
        for variant, row in variants.items():
            if row.get("status") != "success" or "latency_p50_ms" not in row:
                continue
            if row["stage"] == "stage1":
                baseline = "per_q_s1_w8" if row["head_dim"] == 256 else "grouped_s1_allow256_w2"
            elif row["stage"] == "stage2":
                baseline = "unified_s2_w8" if _next_power_of_two(row["head_dim"]) >= 256 else "unified_s2_w4"
            else:
                suffix = "unified_s2_w8" if _next_power_of_two(row["head_dim"]) >= 256 else "unified_s2_w4"
                baseline = ("per_q_s1_w8" if row["head_dim"] == 256 else "grouped_s1_allow256_w2") + "+" + suffix
            baseline_row = variants.get(baseline)
            if baseline_row and baseline_row.get("status") == "success":
                comparisons.append(
                    {
                        "case_id": row["case_id"],
                        "stage": row["stage"],
                        "variant_id": variant,
                        "baseline_id": baseline,
                        "latency_ratio": row["latency_p50_ms"] / baseline_row["latency_p50_ms"],
                        "speedup": baseline_row["latency_p50_ms"] / row["latency_p50_ms"],
                    }
                )
    summaries = {}
    for comparison in comparisons:
        key = (comparison["stage"], comparison["variant_id"], comparison["baseline_id"])
        summaries.setdefault(key, []).append(comparison["latency_ratio"])
    performance_gates = []
    for (stage, variant, baseline), ratios in summaries.items():
        geomean = math.exp(statistics.fmean(math.log(ratio) for ratio in ratios))
        max_ratio = max(ratios)
        geomean_limit = 1.03
        max_limit = 1.10 if "per_q_s1_w8" in baseline else 1.05
        performance_gates.append(
            {
                "stage": stage,
                "variant_id": variant,
                "baseline_id": baseline,
                "case_count": len(ratios),
                "geomean_latency_ratio": geomean,
                "geomean_speedup": 1 / geomean,
                "max_latency_ratio": max_ratio,
                "passed": geomean <= geomean_limit and max_ratio <= max_limit,
                "geomean_limit": geomean_limit,
                "max_case_limit": max_limit,
            }
        )
    return {
        "status_counts": status_counts,
        "required_case_count": len(results),
        "all_required_success": all(row.get("status") == "success" for row in results),
        "comparisons": comparisons,
        "performance_gates": performance_gates,
    }


def _run_info(args, manifest_hash, ncu_version):
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    sm_clock = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,driver_version,memory.total,clocks.sm,clocks.mem,power.limit,temperature.gpu", "--format=csv,noheader"],
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
        "triton": __import__("triton").__version__,
        "cuda_runtime": torch.version.cuda,
        "ncu": ncu_version,
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "compute_capability": [properties.major, properties.minor],
        "gpu_total_memory": properties.total_memory,
        "gpu_runtime_metadata": sm_clock,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
        "thresholds": {"fp16": {"rtol": 0.005, "atol": 0.005}, "bf16": {"rtol": 0.02, "atol": 0.02}},
        "theoretical_assumptions": {"h100_bf16_tflops": H100_BF16_PEAK_TFLOPS, "h100_hbm_tbps": H100_HBM_TBPS},
    }


def _report(run_info, aggregate):
    lines = [
        "# GQA decode 算子测试与优化报告",
        "",
        "## 实验环境",
        "",
        f"- GPU：{run_info['gpu_name']}，compute capability {run_info['compute_capability']}。",
        f"- Torch / Triton / CUDA：{run_info['torch']} / {run_info['triton']} / {run_info['cuda_runtime']}。",
        f"- Git：`{run_info['git_branch']}` @ `{run_info['git_commit']}`。",
        f"- seed：{run_info['seed']}；manifest hash：`{run_info['case_manifest_hash']}`。",
        "",
        "## 正确性与失败状态",
        "",
        f"本性能 run 的 required case 全部成功：{aggregate['all_required_success']}。状态统计：`{aggregate['status_counts']}`。",
        "数值正确性由独立 GPU pytest 的 FP32 oracle gate 负责，性能 run 只在该 gate 通过后执行。",
        "",
        "## 性能 gate",
        "",
        "| stage | candidate | baseline | case | 几何平均加速 | 最差延迟比 | 通过 |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for gate in aggregate["performance_gates"]:
        lines.append(
            f"| {gate['stage']} | {gate['variant_id']} | {gate['baseline_id']} | {gate['case_count']} | "
            f"{gate['geomean_speedup']:.3f}x | {gate['max_latency_ratio']:.3f} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            "## 候选与资源说明",
            "",
            "grouped schedule 每个 CTA 同时处理一个 KV head 对应的 query-head group，以 `tl.dot` 复用 K/V；"
            "per-Q baseline 每个 CTA 处理一个 query head，FP32 显式 reduction 会重复读取共享 K/V。"
            "variant_id 明确记录 schedule、BLOCK_N、num_warps 和 num_stages；每轮候选只改变一个主要变量。",
            "Triton 3.4 launcher 未提供稳定的 register/spill 元数据，相关字段明确记为 unavailable；实际资源证据保存在 `ncu/`。",
            "",
            "## 限制",
            "",
            "本报告只覆盖 head_dim<=256；不据此承诺 D512。NCU cache-control 结果只解释资源瓶颈，最终延迟以未插桩 CUDA event 为准。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Reproducible GQA decode Triton benchmark")
    parser.add_argument("--suite", default="gqa-decode", choices=["gqa-decode"])
    parser.add_argument("--variants", default=",".join(STAGE1_VARIANTS + STAGE2_VARIANTS))
    parser.add_argument("--stages", default="stage1,stage2,combined")
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--head-dims", default="128,256")
    parser.add_argument("--batch-sizes", default="1,8")
    parser.add_argument("--seq-lens", help="Explicit comma-separated lengths; overrides --seq-profile.")
    parser.add_argument(
        "--seq-profile",
        choices=("standard", "broad"),
        default="standard",
        help="Use regular standard lengths or deterministic boundary/irregular broad lengths.",
    )
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--score-modes", default="none,2d,3d")
    parser.add_argument("--block-seqs", default="512")
    parser.add_argument("--slot-orders", default="ordered,shuffled")
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
        raise RuntimeError("CUDA is required for GQA decode benchmarking.")
    if args.num_kv_heads <= 0 or args.num_heads <= args.num_kv_heads or args.num_heads % args.num_kv_heads:
        raise ValueError(f"Expected GQA heads with Hq>Hkv and Hq%Hkv=0, got {args.num_heads}/{args.num_kv_heads}.")
    if not args.profile_only and (args.warmup < 25 or args.rounds < 5 or args.iterations < 100):
        raise ValueError("Formal timing requires warmup>=25, rounds>=5, and iterations>=100.")
    manifest = build_manifest(args)
    manifest_rows = [asdict(case) for case in manifest]
    manifest_hash = _json_hash(manifest_rows)
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + manifest_hash[:8]
    run_dir = args.output_dir / "gqa_decode" / run_id
    writer = ArtifactWriter(run_dir)
    writer.write_json("case_manifest.json", {"sha256": manifest_hash, "cases": manifest_rows})
    _, ncu_version = _ncu_path()
    run_info = _run_info(args, manifest_hash, ncu_version)
    run_info["run_id"] = run_id
    writer.write_json("run_info.json", run_info)
    results = []
    for index, case in enumerate(manifest, start=1):
        print(f"[{index}/{len(manifest)}] {case.case_id} {case.stage} {case.variant_id}", flush=True)
        results.append(run_case(case, writer, profile_only=args.profile_only))
        torch.cuda.empty_cache()
    aggregate = aggregate_results(results)
    writer.write_json("aggregate_metrics.json", aggregate)
    (run_dir / "report.md").write_text(_report(run_info, aggregate), encoding="utf-8")
    print(f"artifacts={run_dir}")
    if not aggregate["all_required_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
