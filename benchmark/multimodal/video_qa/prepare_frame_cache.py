#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.multimodal.video_qa import streamingbench
from benchmark.multimodal.video_qa.datasets import load_video_qa_rows
from benchmark.multimodal.video_qa.evaluate import DEFAULT_DATASET_DIRS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frame cache for unified video QA benchmarks.")
    parser.add_argument("--benchmark", required=True, choices=sorted(DEFAULT_DATASET_DIRS))
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--annotation_dir", default="")
    parser.add_argument("--annotation_path", default="")
    parser.add_argument("--video_dir", default="")
    parser.add_argument("--subtitle_dir", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frame_cache_dir", default="")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--num_video_frames", type=int, default=32)
    parser.add_argument("--context_seconds", type=float, default=-1.0)
    parser.add_argument("--frame_sampling_backend", default="decord", choices=["decord", "ffmpeg"])
    parser.add_argument("--durations", default="all", help="VideoMME only: short,medium,long,all.")
    parser.add_argument("--domains", default="", help="VideoMME only: optional comma-separated domain filter.")
    parser.add_argument("--use_subtitles", action="store_true", help="VideoMME only; does not affect frame cache keys.")
    parser.add_argument("--allow_missing_videos", action="store_true")
    parser.add_argument("--overwrite_frame_cache", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--manifest_path", default="")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.log_every < 1:
        raise ValueError("--log_every must be >= 1.")
    if args.num_video_frames < 1:
        raise ValueError("--num_video_frames must be >= 1.")
    if args.num_samples < -1:
        raise ValueError("--num_samples must be -1 or a non-negative count.")
    if args.sample_start < 0:
        raise ValueError("--sample_start must be non-negative.")
    if args.context_seconds < 0 and args.context_seconds != -1:
        raise ValueError("--context_seconds must be -1 for full video or a non-negative window.")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def worker_prepare(row: dict[str, Any], args_dict: dict[str, Any]) -> dict[str, Any]:
    worker_args = SimpleNamespace(**args_dict)
    start = time.time()
    try:
        frame_paths = streamingbench.ensure_frame_cache(row, worker_args)
        missing = [str(path) for path in frame_paths if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"Missing/empty frame cache files after extraction: {missing[:5]}")
        cache_dir, _frame_paths, start_seconds, end_seconds = streamingbench.frame_cache_paths_for_row(row, worker_args)
        status = "success"
        error = ""
    except Exception as exc:
        frame_paths = []
        cache_dir = None
        start_seconds = None
        end_seconds = None
        status = "frame_extract_failed"
        error = repr(exc)
    return {
        "status": status,
        "error": error,
        "benchmark": row.get("benchmark"),
        "task": row.get("task"),
        "task_type": row.get("task_type"),
        "question_id": row.get("question_id"),
        "sample_id": row.get("sample_id"),
        "video_path": row.get("video_path"),
        "cache_dir": str(cache_dir) if cache_dir is not None else "",
        "context_start_seconds": start_seconds,
        "context_end_seconds": end_seconds,
        "frame_count": len(frame_paths),
        "frame_bytes": sum(path.stat().st_size for path in frame_paths if path.exists()),
        "elapsed_sec": time.time() - start,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.benchmark = str(args.benchmark).lower()
    if not args.dataset_dir:
        args.dataset_dir = DEFAULT_DATASET_DIRS[args.benchmark]
    args.reuse_frame_cache = not args.overwrite_frame_cache
    args.streamingbench_profile = f"unified_{args.benchmark}"
    args.tasks = "all"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_cache_dir = Path(args.frame_cache_dir) if args.frame_cache_dir else output_dir / "frame_cache"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    args.frame_cache_dir = str(frame_cache_dir)

    rows, dataset_info = load_video_qa_rows(args)
    if not rows:
        raise RuntimeError(f"No rows selected for benchmark={args.benchmark}. dataset_info={dataset_info}")
    if args.benchmark == "mvbench" and args.num_samples == -1 and not args.allow_missing_videos:
        expected = dataset_info.get("expected_row_count")
        if expected is not None and len(rows) != int(expected):
            raise RuntimeError(
                f"MVBench frame-cache prep expected {expected} rows, got {len(rows)}. "
                "Use --allow_missing_videos for the available-media 3800-row shard."
            )

    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "frame_cache_manifest.json"
    records_path = manifest_path.with_suffix(".jsonl")
    if records_path.exists():
        records_path.unlink()

    args_dict = {
        "output_dir": str(output_dir),
        "frame_cache_dir": str(frame_cache_dir),
        "reuse_frame_cache": args.reuse_frame_cache,
        "num_video_frames": args.num_video_frames,
        "context_seconds": args.context_seconds,
        "frame_sampling_backend": args.frame_sampling_backend,
    }
    run_info = {
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "cwd": os.getcwd(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset_info": dataset_info,
        "config": {
            "benchmark": args.benchmark,
            "dataset_dir": args.dataset_dir,
            "output_dir": str(output_dir),
            "frame_cache_dir": str(frame_cache_dir),
            "num_video_frames": args.num_video_frames,
            "context_seconds": args.context_seconds,
            "frame_sampling_backend": args.frame_sampling_backend,
            "num_samples": args.num_samples,
            "sample_start": args.sample_start,
            "allow_missing_videos": args.allow_missing_videos,
            "reuse_frame_cache": args.reuse_frame_cache,
            "workers": args.workers,
        },
    }

    counts: dict[str, int] = {}
    failures: list[dict[str, Any]] = []
    frame_bytes = 0
    started = time.time()
    print(
        f"[prepare] benchmark={args.benchmark} rows={len(rows)} workers={args.workers} "
        f"frame_cache_dir={frame_cache_dir} reuse_frame_cache={args.reuse_frame_cache}",
        flush=True,
    )
    with records_path.open("a", encoding="utf-8") as records_handle:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(worker_prepare, row, args_dict) for row in rows]
            for done_count, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                counts[record["status"]] = counts.get(record["status"], 0) + 1
                frame_bytes += int(record.get("frame_bytes", 0) or 0)
                if record["status"] != "success":
                    failures.append(record)
                records_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if done_count <= 5 or done_count == len(rows) or done_count % args.log_every == 0:
                    rate = done_count / max(time.time() - started, 1e-6)
                    print(
                        f"[prepare] {done_count}/{len(rows)} status={record['status']} "
                        f"qid={record.get('question_id')} elapsed={record['elapsed_sec']:.2f}s "
                        f"rate={rate:.2f} rows/s counts={counts}",
                        flush=True,
                    )

    manifest = {
        **run_info,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": time.time() - started,
        "total_rows": len(rows),
        "status_counts": counts,
        "frame_bytes": frame_bytes,
        "frame_gib": frame_bytes / (1024**3),
        "failure_count": len(failures),
        "failure_examples": failures[:10],
        "records_path": str(records_path),
    }
    write_json(manifest_path, manifest)
    if failures:
        raise RuntimeError(f"Frame-cache preparation had {len(failures)} failures. See {manifest_path}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
