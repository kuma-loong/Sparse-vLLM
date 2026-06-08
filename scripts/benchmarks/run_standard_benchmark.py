#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT))
sys.path.insert(0, str(REPO_ROOT_FOR_IMPORT / "src"))

from benchmark.common.ledger import (
    append_ledger_record,
    default_ledger_paths,
    git_metadata,
    selected_env_snapshot,
)
from benchmark.common.paths import (
    REPO_ROOT,
    benchmark_output_root,
    longbench_data_root,
    scbench_preprocessed_root,
)


@dataclass(frozen=True)
class BenchmarkJob:
    name: str
    tier: str
    source: str
    command: list[str]
    output_dir: Path
    dataset: str
    sample_policy: str
    lengths: list[int] | None = None
    max_new_tokens: int | None = None
    required_data: Path | None = None


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _split_ints(value: str) -> list[int]:
    return [int(part) for part in _split_csv(value)]


def _json_arg(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    value = value.strip()
    if value.startswith("@"):
        value = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON argument must decode to an object.")
    return parsed


def _json_cli(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def _method_config(args: argparse.Namespace) -> dict[str, Any]:
    config = _json_arg(args.hyper_param)
    if args.deltakv_checkpoint_path:
        config.setdefault("deltakv_checkpoint_path", args.deltakv_checkpoint_path)
    return config


def _job_output(root: Path, name: str) -> Path:
    return root / name


def _missing_data_marker(name: str) -> Path:
    return Path(f"/__missing_svllm_{name}_data_root__")


def build_jobs(args: argparse.Namespace, run_root: Path) -> list[BenchmarkJob]:
    jobs: list[BenchmarkJob] = []
    selected = set(_split_csv(args.benchmarks))
    methods = ",".join(_split_csv(args.methods))
    method_config = _method_config(args)

    def wanted(name: str) -> bool:
        return "all" in selected or name in selected

    if wanted("sanity"):
        out = _job_output(run_root, "sanity")
        cmd = _python_cmd(
            "benchmark/sanity.py",
            "--model_path",
            args.model_path,
            "--sparse_method",
            args.primary_method,
            "--backend",
            args.backend,
            "--cuda_device",
            str(args.cuda_device),
            "--max_new_tokens",
            str(args.sanity_max_new_tokens),
            "--output_dir",
            str(out),
        )
        if args.tokenizer_path:
            cmd.extend(["--tokenizer_path", args.tokenizer_path])
        if args.deltakv_checkpoint_path:
            cmd.extend(["--deltakv_checkpoint_path", args.deltakv_checkpoint_path])
        if method_config:
            cmd.extend(["--hyper_param", _json_cli(method_config)])
        jobs.append(
            BenchmarkJob(
                name="sanity",
                tier="Tier 0",
                source="repo_existing",
                command=cmd,
                output_dir=out,
                dataset="inline_prompts",
                sample_policy="smoke",
                max_new_tokens=args.sanity_max_new_tokens,
            )
        )

    if wanted("microbench"):
        if args.microbench_output_len is None:
            selected_lengths = _split_ints(args.lengths)
            unsupported_lengths = [
                length for length in selected_lengths if 16000 < length < 32000
            ]
            if unsupported_lengths:
                raise ValueError(
                    "Default microbench output length is only fixed for <=16k and >=32k contexts. "
                    f"Unsupported lengths: {unsupported_lengths}. Pass --microbench_output_len explicitly."
                )
            microbench_specs = [
                ("microbench_decode512", [length for length in selected_lengths if length <= 16000], 512),
                ("microbench_decode1024", [length for length in selected_lengths if length >= 32000], 1024),
            ]
        else:
            microbench_specs = [("microbench", _split_ints(args.lengths), args.microbench_output_len)]

        for job_name, job_lengths, output_len in microbench_specs:
            if not job_lengths:
                continue
            out = _job_output(run_root, job_name)
            cmd = _python_cmd(
                "scripts/benchmarks/bench_sparse_vllm.py",
                "--model_path",
                args.model_path,
                "--methods",
                methods,
                "--lengths",
                ",".join(str(length) for length in job_lengths),
                "--batch_sizes",
                args.batch_sizes,
                "--output_len",
                str(output_len),
                "--output_dir",
                str(out),
            )
            if method_config:
                cmd.extend(["--hyper_params", _json_cli(method_config)])
            jobs.append(
                BenchmarkJob(
                    name=job_name,
                    tier="Tier 0",
                    source="repo_existing",
                    command=cmd,
                    output_dir=out,
                    dataset="synthetic",
                    sample_policy=args.sample_policy,
                    lengths=job_lengths,
                    max_new_tokens=output_len,
                )
            )

    if wanted("niah"):
        out = _job_output(run_root, "niah")
        cmd = _python_cmd(
            "benchmark/niah/test_niah.py",
            "--model_path",
            args.model_path,
            "--output_path",
            str(out),
            "--online_test",
            "True",
            "--context_lengths",
            args.niah_context_lengths,
            "--max_new_tokens",
            str(args.niah_max_new_tokens),
            "--sparse_method",
            args.primary_method,
            "--backend",
            args.backend,
            "--cuda_device",
            str(args.cuda_device),
        )
        if args.tokenizer_path:
            cmd.extend(["--tokenizer_path", args.tokenizer_path])
        if args.deltakv_checkpoint_path:
            cmd.extend(["--deltakv_checkpoint_path", args.deltakv_checkpoint_path])
        jobs.append(
            BenchmarkJob(
                name="niah",
                tier="Tier 1",
                source="repo_existing",
                command=cmd,
                output_dir=out,
                dataset="synthetic_niah",
                sample_policy=args.sample_policy,
                lengths=[value * 1000 for value in _split_ints(args.niah_context_lengths)],
                max_new_tokens=args.niah_max_new_tokens,
            )
        )

    if wanted("longbench"):
        data_root = Path(args.longbench_data_root).expanduser() if args.longbench_data_root else longbench_data_root()
        out = _job_output(run_root, "longbench")
        cmd = _python_cmd(
            "benchmark/long_bench/pred.py",
            "--model",
            args.model_name,
            "--model_path",
            args.model_path,
            "--sparse_method",
            args.primary_method,
            "--backend",
            args.backend,
            "--task",
            args.longbench_tasks,
            "--num_samples",
            str(args.longbench_num_samples),
            "--batch_size",
            str(args.longbench_batch_size),
            "--output_root",
            str(out),
        )
        if args.longbench_e:
            cmd.append("--e")
        if args.tokenizer_path:
            cmd.extend(["--tokenizer_path", args.tokenizer_path])
        if args.deltakv_checkpoint_path:
            cmd.extend(["--deltakv_checkpoint_path", args.deltakv_checkpoint_path])
        if method_config:
            cmd.extend(["--hyper_param", _json_cli(method_config)])
        jobs.append(
            BenchmarkJob(
                name="longbench",
                tier="Tier 2",
                source="repo_existing",
                command=cmd,
                output_dir=out,
                dataset=args.longbench_tasks,
                sample_policy=args.sample_policy,
                required_data=data_root if data_root is not None else _missing_data_marker("longbench"),
            )
        )

    if wanted("scbench"):
        data_root = Path(args.scbench_data_root).expanduser() if args.scbench_data_root else scbench_preprocessed_root()
        out = _job_output(run_root, "scbench")
        scbench_hyper = dict(method_config)
        cmd = _python_cmd(
            "benchmark/scbench/run_scbench_preprocessed.py",
            "--task",
            args.scbench_tasks,
            "--data_root",
            str(data_root or ""),
            "--output_dir",
            str(out),
            "--model_name_or_path",
            args.model_path,
            "--attn_type",
            args.scbench_attn_type or args.primary_method,
            "--kv_type",
            args.scbench_kv_type,
            "--num_eval_examples",
            str(args.scbench_num_examples),
            "--max_seq_length",
            str(args.scbench_max_seq_length),
            "--hyper_param",
            _json_cli(scbench_hyper),
        )
        jobs.append(
            BenchmarkJob(
                name="scbench",
                tier="Tier 3",
                source="repo_existing",
                command=cmd,
                output_dir=out,
                dataset=args.scbench_tasks,
                sample_policy=args.sample_policy,
                required_data=data_root if data_root is not None else _missing_data_marker("scbench"),
            )
        )

    if wanted("mathbench"):
        out = _job_output(run_root, "mathbench")
        cmd = _python_cmd(
            "benchmark/math_bench/pred.py",
            "--model",
            args.model_name,
            "--model_path",
            args.model_path,
            "--sparse_method",
            args.primary_method,
            "--backend",
            args.backend,
            "--task",
            args.math_tasks,
            "--num_samples",
            str(args.math_num_samples),
            "--max_new_tokens",
            str(args.math_max_new_tokens),
            "--temperature",
            str(args.math_temperature),
            "--batch_size",
            str(args.math_batch_size),
            "--output_root",
            str(out),
        )
        if args.tokenizer_path:
            cmd.extend(["--tokenizer_path", args.tokenizer_path])
        if args.deltakv_checkpoint_path:
            cmd.extend(["--deltakv_checkpoint_path", args.deltakv_checkpoint_path])
        if method_config:
            cmd.extend(["--hyper_param", _json_cli(method_config)])
        jobs.append(
            BenchmarkJob(
                name="mathbench",
                tier="Tier 4",
                source="repo_existing",
                command=cmd,
                output_dir=out,
                dataset=args.math_tasks,
                sample_policy=args.sample_policy,
                max_new_tokens=args.math_max_new_tokens,
            )
        )

    return jobs


def _read_metrics(output_dir: Path) -> dict[str, Any]:
    candidates = [
        output_dir / "aggregate_metrics.json",
        output_dir / "result.json",
        output_dir / "last_benchmark_result.json",
    ]
    if output_dir.exists():
        candidates.extend(sorted(output_dir.glob("*/result.json")))
        candidates.extend(sorted(output_dir.glob("*/*/result.json")))
    for candidate in candidates:
        if candidate.exists():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                return payload if isinstance(payload, dict) else {"result": payload}
            except Exception as exc:
                return {"metric_read_error": repr(exc), "metric_path": str(candidate)}
    return {}


def _status_from_returncode(returncode: int, metrics: dict[str, Any]) -> str:
    if returncode == 0:
        metric_status = metrics.get("status")
        if isinstance(metric_status, str) and metric_status in {"model_failed", "metric_failed", "parse_failed"}:
            return metric_status
        return "success"
    return "model_failed"


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def run_job(args: argparse.Namespace, job: BenchmarkJob, ledger_jsonl: Path, ledger_csv: Path, index: int) -> int:
    if job.required_data is not None and not job.required_data.is_dir():
        status = "invalid_run"
        failure_summary = f"Required data directory is missing: {job.required_data}"
        returncode = 2
        elapsed_s = 0.0
        metrics: dict[str, Any] = {}
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "failure.json").write_text(
            json.dumps({"status": status, "failure_summary": failure_summary}, indent=2) + "\n",
            encoding="utf-8",
        )
    else:
        job.output_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = job.output_dir / "stdout.log"
        stderr_path = job.output_dir / "stderr.log"
        env = os.environ.copy()
        src_path = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = f"{src_path}:{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else src_path
        if args.use_proxy_7890:
            env.setdefault("http_proxy", "http://localhost:7890")
            env.setdefault("https_proxy", "http://localhost:7890")

        started = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            proc = subprocess.run(job.command, cwd=REPO_ROOT, env=env, text=True, stdout=stdout, stderr=stderr)
        elapsed_s = time.perf_counter() - started
        metrics = _read_metrics(job.output_dir)
        status = _status_from_returncode(proc.returncode, metrics)
        failure_summary = "" if proc.returncode == 0 else f"exit code {proc.returncode}; see {stderr_path}"
        returncode = proc.returncode

    git = git_metadata(REPO_ROOT)
    run_id = f"{args.feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{git['git_commit']}_{index:03d}_{job.name}"
    record = {
        "run_id": run_id,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "feature": args.feature,
        "objective": args.objective,
        **git,
        "benchmark": job.name,
        "benchmark_tier": job.tier,
        "benchmark_source": job.source,
        "script": job.command[1] if len(job.command) > 1 else job.command[0],
        "command": _shell_join(job.command),
        "model_path": args.model_path,
        "tokenizer_path": args.tokenizer_path,
        "method": args.primary_method,
        "method_config": _method_config(args),
        "baseline_run_id": args.baseline_run_id,
        "previous_run_id": args.previous_run_id,
        "dataset": job.dataset,
        "split": "test",
        "sample_policy": job.sample_policy,
        "sample_ids": args.sample_ids,
        "lengths": job.lengths,
        "max_new_tokens": job.max_new_tokens,
        "decode_config": {
            "microbench_output_len": job.max_new_tokens if job.name.startswith("microbench") else args.microbench_output_len,
            "math_temperature": args.math_temperature,
        },
        "gpu": os.getenv("CUDA_VISIBLE_DEVICES", f"cuda_device={args.cuda_device}"),
        "env": selected_env_snapshot(),
        "output_dir": str(job.output_dir),
        "status": status,
        "primary_metrics": metrics,
        "quality_delta": None,
        "speedup": None,
        "memory_delta": None,
        "failure_summary": failure_summary,
        "decision": "keep" if status == "success" else "investigate",
        "notes": f"elapsed_s={elapsed_s:.3f}; returncode={returncode}",
    }
    append_ledger_record(record, jsonl_path=ledger_jsonl, csv_path=ledger_csv)
    print(json.dumps({"job": job.name, "status": status, "output_dir": str(job.output_dir)}, ensure_ascii=False))
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sparse-vLLM standard quick/final benchmark plans.")
    parser.add_argument("--mode", choices=["quick", "final"], default="quick")
    parser.add_argument("--feature", required=True, help="Feature or optimization slug used for ledger files.")
    parser.add_argument("--objective", default="")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="model")
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--backend", default="hf", choices=["hf", "sparsevllm"])
    parser.add_argument("--primary_method", default="vanilla")
    parser.add_argument("--methods", default="vanilla")
    parser.add_argument("--deltakv_checkpoint_path", default=None)
    parser.add_argument("--hyper_param", default=None, help="Inline JSON or @path JSON object.")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--ledger_jsonl", default=None)
    parser.add_argument("--ledger_csv", default=None)
    parser.add_argument("--baseline_run_id", default=None)
    parser.add_argument("--previous_run_id", default=None)
    parser.add_argument("--sample_ids", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--continue_on_failure", action="store_true")
    parser.add_argument("--use_proxy_7890", action="store_true")

    parser.add_argument("--benchmarks", default=None, help="Comma-separated benchmark names or all.")
    parser.add_argument("--sample_policy", default=None)

    parser.add_argument("--sanity_max_new_tokens", type=int, default=16)
    parser.add_argument("--lengths", default=None)
    parser.add_argument("--batch_sizes", default="1")
    parser.add_argument("--microbench_output_len", type=int, default=None)
    parser.add_argument("--niah_context_lengths", default=None)
    parser.add_argument("--niah_max_new_tokens", type=int, default=20)
    parser.add_argument("--longbench_data_root", default=None)
    parser.add_argument("--longbench_tasks", default=None)
    parser.add_argument("--longbench_num_samples", type=int, default=None)
    parser.add_argument("--longbench_batch_size", type=int, default=1)
    parser.add_argument("--longbench_e", action="store_true")
    parser.add_argument("--scbench_data_root", default=None)
    parser.add_argument("--scbench_tasks", default=None)
    parser.add_argument("--scbench_attn_type", default=None)
    parser.add_argument("--scbench_kv_type", default="dense")
    parser.add_argument("--scbench_num_examples", type=int, default=None)
    parser.add_argument("--scbench_max_seq_length", type=int, default=131072)
    parser.add_argument("--math_tasks", default="gsm8k")
    parser.add_argument("--math_num_samples", type=int, default=None)
    parser.add_argument("--math_max_new_tokens", type=int, default=None)
    parser.add_argument("--math_temperature", type=float, default=0.6)
    parser.add_argument("--math_batch_size", type=int, default=1)
    return parser.parse_args()


def apply_mode_defaults(args: argparse.Namespace) -> None:
    if args.benchmarks is None:
        args.benchmarks = "sanity,microbench,niah,longbench" if args.mode == "quick" else "sanity,microbench,niah,scbench,longbench,mathbench"
    if args.sample_policy is None:
        args.sample_policy = "smoke" if args.mode == "quick" else "full"
    if args.lengths is None:
        args.lengths = "1024,4096,16000,32000,64000,128000,256000"
    if args.niah_context_lengths is None:
        args.niah_context_lengths = "16,32" if args.mode == "quick" else "16,32,64,128"
    if args.longbench_tasks is None:
        args.longbench_tasks = "qasper,hotpotqa,passage_retrieval_en" if args.mode == "quick" else "narrativeqa,qasper,hotpotqa,2wikimqa,gov_report,qmsum,passage_retrieval_en,passage_count,lcc,repobench-p"
    if args.longbench_num_samples is None:
        args.longbench_num_samples = 20 if args.mode == "quick" else -1
    if args.scbench_tasks is None:
        args.scbench_tasks = "scbench_kv" if args.mode == "quick" else "scbench_kv,scbench_qa_eng,scbench_summary_with_needles"
    if args.scbench_num_examples is None:
        args.scbench_num_examples = 20 if args.mode == "quick" else -1
    if args.math_num_samples is None:
        args.math_num_samples = 50 if args.mode == "quick" else -1
    if args.math_max_new_tokens is None:
        args.math_max_new_tokens = 512 if args.mode == "quick" else 32768


def main() -> None:
    args = parse_args()
    apply_mode_defaults(args)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser() if args.output_root else benchmark_output_root() / "standard" / args.feature / args.mode / time_tag
    output_root.mkdir(parents=True, exist_ok=True)
    ledger_jsonl, ledger_csv = default_ledger_paths(args.feature, benchmark_output_root())
    if args.ledger_jsonl:
        ledger_jsonl = Path(args.ledger_jsonl).expanduser()
    if args.ledger_csv:
        ledger_csv = Path(args.ledger_csv).expanduser()

    jobs = build_jobs(args, output_root)
    plan = {
        "mode": args.mode,
        "feature": args.feature,
        "objective": args.objective,
        "output_root": str(output_root),
        "ledger_jsonl": str(ledger_jsonl),
        "ledger_csv": str(ledger_csv),
        "jobs": [{"name": job.name, "command": _shell_join(job.command), "output_dir": str(job.output_dir)} for job in jobs],
    }
    (output_root / "benchmark_plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False))

    if args.dry_run:
        return

    failures = []
    for index, job in enumerate(jobs, start=1):
        code = run_job(args, job, ledger_jsonl, ledger_csv, index)
        if code != 0:
            failures.append((job.name, code))
            if not args.continue_on_failure:
                break

    summary = {"failures": failures, "num_jobs": len(jobs), "status": "success" if not failures else "failed"}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
