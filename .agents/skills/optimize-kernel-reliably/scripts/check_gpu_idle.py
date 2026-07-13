#!/usr/bin/env python3
"""Select an idle NVIDIA GPU without hiding unavailable device telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


GPU_FIELDS = (
    "index",
    "uuid",
    "name",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "pstate",
    "mig.mode.current",
)


def _number(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
        return None
    return float(value)


def parse_gpu_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for values in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(values) != len(GPU_FIELDS):
            raise ValueError(f"expected {len(GPU_FIELDS)} GPU fields, got {len(values)}: {values}")
        row = dict(zip(GPU_FIELDS, (value.strip() for value in values)))
        rows.append(
            {
                "index": int(row["index"]),
                "uuid": row["uuid"],
                "name": row["name"],
                "utilization_gpu_pct": _number(row["utilization.gpu"]),
                "memory_used_mib": _number(row["memory.used"]),
                "memory_total_mib": _number(row["memory.total"]),
                "temperature_c": _number(row["temperature.gpu"]),
                "power_draw_w": _number(row["power.draw"]),
                "clock_sm_mhz": _number(row["clocks.sm"]),
                "clock_memory_mhz": _number(row["clocks.mem"]),
                "pstate": row["pstate"],
                "mig_mode": row["mig.mode.current"],
                "processes": [],
            }
        )
    return rows


def parse_process_rows(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for values in csv.reader(line for line in output.splitlines() if line.strip()):
        if values and values[0].strip().lower().startswith("no running"):
            continue
        if len(values) != 4:
            raise ValueError(f"expected 4 compute-process fields, got {len(values)}: {values}")
        gpu_uuid, pid, process_name, memory = (value.strip() for value in values)
        rows.append(
            {
                "gpu_uuid": gpu_uuid,
                "pid": int(pid),
                "process_name": process_name,
                "used_gpu_memory_mib": _number(memory),
            }
        )
    return rows


def _run(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def _visible_gpus(gpus: list[dict[str, object]]) -> list[dict[str, object]]:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES")
    if selector is None:
        visible = list(gpus)
    else:
        tokens = [token.strip() for token in selector.split(",") if token.strip()]
        if not tokens:
            return []
        by_index = {str(gpu["index"]): gpu for gpu in gpus}
        by_uuid = {str(gpu["uuid"]): gpu for gpu in gpus}
        visible = []
        for token in tokens:
            gpu = by_index.get(token) or by_uuid.get(token)
            if gpu is None:
                matches = [row for uuid, row in by_uuid.items() if uuid.startswith(token)]
                if len(matches) != 1:
                    raise ValueError(f"CUDA_VISIBLE_DEVICES entry {token!r} does not identify one GPU")
                gpu = matches[0]
            if gpu not in visible:
                visible.append(gpu)
    for visible_ordinal, gpu in enumerate(visible):
        gpu["visible_ordinal"] = visible_ordinal
    return visible


def take_snapshot(nvidia_smi: str, max_utilization: float, max_memory_used_mib: float) -> dict[str, object]:
    gpu_output = _run(
        [
            nvidia_smi,
            f"--query-gpu={','.join(GPU_FIELDS)}",
            "--format=csv,noheader,nounits",
        ]
    )
    process_output = _run(
        [
            nvidia_smi,
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = parse_gpu_rows(gpu_output)
    processes = parse_process_rows(process_output)
    by_uuid = {str(gpu["uuid"]): gpu for gpu in gpus}
    for process in processes:
        gpu = by_uuid.get(str(process["gpu_uuid"]))
        if gpu is not None:
            gpu["processes"].append(process)

    visible = _visible_gpus(gpus)
    for gpu in visible:
        gpu["mps_process_detected"] = any(
            "mps" in str(process["process_name"]).lower() for process in gpu["processes"]
        )
        reasons = []
        utilization = gpu["utilization_gpu_pct"]
        memory_used = gpu["memory_used_mib"]
        if utilization is None:
            reasons.append("utilization unavailable")
        elif utilization > max_utilization:
            reasons.append(f"utilization {utilization:g}% > {max_utilization:g}%")
        if memory_used is None:
            reasons.append("memory usage unavailable")
        elif memory_used > max_memory_used_mib:
            reasons.append(f"memory {memory_used:g} MiB > {max_memory_used_mib:g} MiB")
        if gpu["processes"]:
            reasons.append(f"{len(gpu['processes'])} compute process(es)")
        gpu["idle"] = not reasons
        gpu["busy_reasons"] = reasons

    idle = [gpu for gpu in visible if gpu["idle"]]
    selected = min(idle, key=lambda row: int(row["visible_ordinal"])) if idle else None
    return {
        "schema_version": "1.0",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "admission": {
            "max_utilization_gpu_pct": max_utilization,
            "max_memory_used_mib": max_memory_used_mib,
            "require_no_compute_processes": True,
        },
        "gpus": visible,
        "selected": selected,
        "all_devices_busy": selected is None,
    }


def _write_result(result: dict[str, object], output: Path | None) -> None:
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        sys.stdout.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--max-utilization", type=float, default=0.0)
    parser.add_argument("--max-memory-used-mib", type=float, default=512.0)
    parser.add_argument("--wait-seconds", type=float, default=0.0)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_utilization < 0 or args.max_memory_used_mib < 0:
        parser.error("admission thresholds must be non-negative")
    if args.wait_seconds < 0 or args.poll_seconds <= 0:
        parser.error("wait must be non-negative and poll must be positive")

    deadline = time.monotonic() + args.wait_seconds
    while True:
        try:
            result = take_snapshot(args.nvidia_smi, args.max_utilization, args.max_memory_used_mib)
        except (OSError, RuntimeError, ValueError) as exc:
            _write_result(
                {
                    "schema_version": "1.0",
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc),
                },
                args.output,
            )
            return 2
        if result["selected"] is not None or time.monotonic() >= deadline:
            _write_result(result, args.output)
            return 0 if result["selected"] is not None else 1
        remaining = deadline - time.monotonic()
        time.sleep(min(args.poll_seconds, remaining, 60.0))


if __name__ == "__main__":
    raise SystemExit(main())
