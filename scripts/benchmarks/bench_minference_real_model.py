from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import time
from pathlib import Path
from time import perf_counter
from typing import Any


METHOD_CONFIGS = {
    "vanilla": {
        "sparse_method": "vanilla",
    },
    "snapkv": {
        "sparse_method": "snapkv",
    },
    "minference_full": {
        "sparse_method": "vanilla",
        "prefill_attention_backend": "minference",
    },
    "minference_snapkv": {
        "sparse_method": "snapkv",
        "prefill_attention_backend": "minference",
    },
}


def _load_json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    value = str(value).strip()
    if value.startswith("@"):
        value = Path(value[1:]).read_text(encoding="utf-8")
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("--hyper_params must be a JSON object.")
    return data


def _gpu_is_idle(gpu_index: str, max_used_mb: int, max_util: int) -> tuple[bool, str]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in out.strip().splitlines():
        idx, used, util = [part.strip() for part in line.split(",")]
        if idx != str(gpu_index):
            continue
        used_i = int(used)
        util_i = int(util)
        ok = used_i <= int(max_used_mb) and util_i <= int(max_util)
        return ok, f"gpu={idx} used_mb={used_i} util_pct={util_i}"
    return False, f"gpu={gpu_index} not found"


def _wait_for_idle_gpu(args) -> tuple[bool, str]:
    deadline = time.time() + float(args.idle_wait_s)
    last_state = ""
    while True:
        ok, state = _gpu_is_idle(args.gpu, args.max_gpu_used_mb, args.max_gpu_util)
        last_state = state
        if ok:
            return True, state
        if time.time() >= deadline:
            return False, last_state
        time.sleep(float(args.idle_poll_s))


def _resolve_master_port(args) -> int:
    if args.master_port is not None:
        return int(args.master_port)
    try:
        return 2333 + int(str(args.gpu).split(",")[0])
    except ValueError:
        return 2333


def _build_engine_kwargs(args, method: str, prompt_len: int) -> dict[str, Any]:
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unknown method {method!r}. Valid methods: {sorted(METHOD_CONFIGS)}")

    chunk_size = int(prompt_len)
    kwargs: dict[str, Any] = {
        "enforce_eager": True,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "tensor_parallel_size": 1,
        "engine_prefill_chunk_size": chunk_size,
        "max_num_batched_tokens": max(2 * chunk_size + 16, chunk_size + int(args.output_len) + 16),
        "max_num_seqs_in_batch": int(args.batch_size),
        "max_decoding_seqs": int(args.batch_size),
        "max_model_len": int(prompt_len) + int(args.output_len) + 64,
        "throughput_log_interval_s": 0.0,
        "sink_keep_tokens": int(args.sink_keep_tokens),
        "recent_keep_tokens": int(args.recent_keep_tokens),
        "decode_keep_tokens": int(args.decode_keep_tokens),
        "prefill_keep_tokens": int(args.prefill_keep_tokens),
    }
    extra_kwargs = _load_json_arg(args.hyper_params)
    kwargs.update(extra_kwargs)
    kwargs.update(METHOD_CONFIGS[method])
    if method.startswith("minference"):
        kwargs.setdefault("minference_config_path", str(args.minference_config_path))
        kwargs.setdefault("minference_starting_layer", int(args.minference_starting_layer))
        kwargs.setdefault("minference_ratio", float(args.minference_ratio))
    return kwargs


def _run_case(args, method: str, prompt_len: int) -> dict[str, Any]:
    import torch

    from sparsevllm import LLM, SamplingParams
    from sparsevllm.utils.profiler import profiler

    idle_ok, gpu_state = _wait_for_idle_gpu(args)
    if not idle_ok:
        raise RuntimeError(f"Refusing to run on non-idle GPU: {gpu_state}")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    engine_kwargs = _build_engine_kwargs(args, method, prompt_len)
    llm = None
    result: dict[str, Any] = {
        "method": method,
        "prompt_len": int(prompt_len),
        "batch_size": int(args.batch_size),
        "output_len": int(args.output_len),
        "engine_kwargs": engine_kwargs,
        "gpu_state_before": gpu_state,
        "status": "started",
    }
    try:
        load_start = perf_counter()
        llm = LLM(args.model_path, **engine_kwargs)
        torch.cuda.synchronize()
        result["engine_load_s"] = perf_counter() - load_start

        prompt_token_ids = [[int(args.prompt_token_id)] * int(prompt_len) for _ in range(int(args.batch_size))]
        sampling_params = [
            SamplingParams(
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
                ignore_eos=True,
                max_tokens=int(args.output_len),
            )
            for _ in range(int(args.batch_size))
        ]
        for prompt, params in zip(prompt_token_ids, sampling_params):
            llm.add_request(prompt, params)

        profiler.reset()
        torch.cuda.synchronize()
        start = perf_counter()
        ttft_s: float | None = None
        prefill_s = 0.0
        decode_s = 0.0
        prefill_tokens = 0
        decode_tokens = 0
        step_count = 0
        first_completion_tokens = 0
        completion_tokens_by_seq: dict[int, int] = {}

        while not llm.is_finished():
            step_start = perf_counter()
            finished_outputs, num_tokens = llm.step()
            torch.cuda.synchronize()
            step_s = perf_counter() - step_start
            step_count += 1

            if num_tokens > 0:
                prefill_tokens += int(num_tokens)
                prefill_s += step_s
                current_outputs = getattr(llm, "last_step_token_outputs", [])
                if ttft_s is None and current_outputs:
                    ttft_s = perf_counter() - start
                    first_completion_tokens += sum(len(toks) for _, toks in current_outputs)
            elif num_tokens < 0:
                decode_tokens += int(-num_tokens)
                decode_s += step_s
                if ttft_s is None:
                    ttft_s = perf_counter() - start
            for seq_id, token_ids in finished_outputs:
                completion_tokens_by_seq[int(seq_id)] = len(token_ids)

        total_s = perf_counter() - start
        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
        generated_tokens = sum(completion_tokens_by_seq.values())
        measured_decode_tokens = decode_tokens
        tpot_ms = (decode_s / measured_decode_tokens * 1000.0) if measured_decode_tokens > 0 else 0.0
        result.update(
            {
                "status": "success",
                "total_s": total_s,
                "ttft_s": float(ttft_s or 0.0),
                "tpot_ms": tpot_ms,
                "prefill_s": prefill_s,
                "decode_s": decode_s,
                "prefill_tokens": prefill_tokens,
                "decode_tokens": decode_tokens,
                "generated_tokens": generated_tokens,
                "first_completion_tokens": first_completion_tokens,
                "prefill_tok_s": prefill_tokens / prefill_s if prefill_s > 0 else 0.0,
                "decode_tok_s": measured_decode_tokens / decode_s if decode_s > 0 else 0.0,
                "peak_mem_gb": peak_mem_gb,
                "step_count": step_count,
            }
        )
    except Exception as exc:
        result.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if llm is not None:
            llm.exit()
        torch.cuda.empty_cache()
    return result


def _run_case_worker(args, method: str, prompt_len: int, queue: mp.Queue):
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["SPARSEVLLM_MASTER_PORT"] = str(_resolve_master_port(args))
    try:
        queue.put(_run_case(args, method, prompt_len))
    except Exception as exc:
        queue.put(
            {
                "method": method,
                "prompt_len": int(prompt_len),
                "batch_size": int(args.batch_size),
                "output_len": int(args.output_len),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def _run_case_in_subprocess(args, method: str, prompt_len: int) -> dict[str, Any]:
    idle_ok, gpu_state = _wait_for_idle_gpu(args)
    if not idle_ok:
        return {
            "method": method,
            "prompt_len": int(prompt_len),
            "batch_size": int(args.batch_size),
            "output_len": int(args.output_len),
            "status": "failed",
            "error": f"RuntimeError: Refusing to run on non-idle GPU before spawn: {gpu_state}",
        }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_run_case_worker, args=(args, method, prompt_len, queue))
    process.start()
    process.join(timeout=float(args.case_timeout_s))
    if process.is_alive():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.join(timeout=10.0)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.join(timeout=10.0)
        return {
            "method": method,
            "prompt_len": int(prompt_len),
            "batch_size": int(args.batch_size),
            "output_len": int(args.output_len),
            "status": "timeout",
            "error": f"case exceeded timeout_s={float(args.case_timeout_s):.1f}",
        }
    if not queue.empty():
        row = queue.get()
    else:
        row = {
            "method": method,
            "prompt_len": int(prompt_len),
            "batch_size": int(args.batch_size),
            "output_len": int(args.output_len),
            "status": "failed",
            "error": f"worker exited without result, exitcode={process.exitcode}",
        }
    if process.exitcode not in (0, None) and row.get("status") == "success":
        row["status"] = "failed"
        row["error"] = f"worker exitcode={process.exitcode}"
    return row


def _write_report(output_dir: Path, results: list[dict[str, Any]], args):
    report_path = output_dir / "report.md"
    first_success_kwargs = next(
        (row.get("engine_kwargs", {}) for row in results if row.get("status") == "success"),
        {},
    )
    pattern_config = first_success_kwargs.get("minference_config_path", str(args.minference_config_path))
    lines = [
        "# Qwen2.5-7B-Instruct-1M MInference Benchmark",
        "",
        f"- Model: `{args.model_path}`",
        f"- Pattern config: `{pattern_config}`",
        f"- GPU: `CUDA_VISIBLE_DEVICES={args.gpu}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Output length: `{args.output_len}`",
        f"- Case timeout: `{args.case_timeout_s}s`",
        f"- SPARSEVLLM_MASTER_PORT: `{_resolve_master_port(args)}`",
        f"- Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- Command: `{' '.join(os.sys.argv)}`",
        "",
        "| Method | Prompt tokens | Status | TTFT s | TPOT ms | Prefill tok/s | Decode tok/s | Peak GB | MInf ratio | Total s |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        if row.get("status") != "success":
            lines.append(
                f"| {row.get('method')} | {row.get('prompt_len')} | {row.get('status')}: {row.get('error', '')} | | | | | | | |"
            )
            continue
        engine_kwargs = row.get("engine_kwargs", {})
        row = dict(row)
        row["minference_ratio_effective"] = engine_kwargs.get("minference_ratio", "")
        lines.append(
            "| {method} | {prompt_len} | success | {ttft_s:.3f} | {tpot_ms:.3f} | "
            "{prefill_tok_s:.1f} | {decode_tok_s:.1f} | {peak_mem_gb:.2f} | "
            "{minference_ratio_effective} | {total_s:.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Raw per-case records are stored in `results.jsonl`. Failed rows are not interpolated or estimated.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/data2/guquansheng/Qwen2.5-7B-Instruct-1M")
    parser.add_argument(
        "--minference_config_path",
        default="reference/MInference/minference/configs/Qwen2.5_7B_Instruct_1M.json",
    )
    parser.add_argument("--methods", default="vanilla,snapkv,minference_full,minference_snapkv")
    parser.add_argument("--lengths", default="2048,8192,32768")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_len", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--max_gpu_used_mb", type=int, default=1024)
    parser.add_argument("--max_gpu_util", type=int, default=5)
    parser.add_argument("--idle_wait_s", type=float, default=120.0)
    parser.add_argument("--idle_poll_s", type=float, default=5.0)
    parser.add_argument("--case_timeout_s", type=float, default=900.0)
    parser.add_argument("--master_port", type=int, default=None)
    parser.add_argument("--allow_failures", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.86)
    parser.add_argument("--prompt_token_id", type=int, default=100)
    parser.add_argument("--sink_keep_tokens", type=int, default=64)
    parser.add_argument("--recent_keep_tokens", type=int, default=512)
    parser.add_argument("--decode_keep_tokens", type=int, default=4096)
    parser.add_argument("--prefill_keep_tokens", type=int, default=8192)
    parser.add_argument("--minference_starting_layer", type=int, default=0)
    parser.add_argument("--minference_ratio", type=float, default=1.0)
    parser.add_argument("--hyper_params", default="{}")
    parser.add_argument("--output_dir", default="benchmark/results/minference_qwen25_7b_1m")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    lengths = [int(item.strip()) for item in args.lengths.split(",") if item.strip()]

    results: list[dict[str, Any]] = []
    with results_path.open("a", encoding="utf-8") as f:
        for prompt_len in lengths:
            for method in methods:
                print(f"[case] method={method} prompt_len={prompt_len}", flush=True)
                row = _run_case_in_subprocess(args, method, prompt_len)
                if row.get("status") != "success":
                    print(f"[failed] {row.get('error', '')}", flush=True)
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                f.flush()
                _write_report(output_dir, results, args)
    print(f"[done] report={output_dir / 'report.md'} results={results_path}", flush=True)
    if any(row.get("status") != "success" for row in results) and not args.allow_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
