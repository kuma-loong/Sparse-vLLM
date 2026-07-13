#!/usr/bin/env python3
"""Compare paired baseline and candidate kernel latency samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import statistics
import sys


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def _latency(row: dict[str, object]) -> float:
    samples = row.get("latency_samples_ms")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{row.get('comparison_id')}: latency_samples_ms must be a non-empty list")
    values = [float(value) for value in samples]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{row.get('comparison_id')}: latency samples must be positive and finite")
    return statistics.median(values)


def _weighted_geomean(ratios: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("comparison weights must sum to a positive value")
    return math.exp(sum(weight * math.log(ratio) for ratio, weight in zip(ratios, weights)) / total_weight)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = quantile * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def compare(
    rows: list[dict[str, object]],
    baseline: str,
    candidate: str,
    confidence_level: float,
    bootstrap_resamples: int,
    seed: int,
    minimum_improvement: float,
    maximum_case_regression: float,
) -> dict[str, object]:
    selected = [row for row in rows if row.get("variant_id") in {baseline, candidate}]
    indexed: dict[tuple[str, str], dict[str, object]] = {}
    for row in selected:
        required_fields = (
            "schema_version",
            "comparison_id",
            "case_id",
            "variant_id",
            "latency_samples_ms",
            "pair_order",
            "status",
        )
        for field in required_fields:
            if field not in row:
                raise ValueError(f"formal row is missing {field}: {row}")
        if row["schema_version"] != "1.0":
            raise ValueError(f"unsupported schema_version: {row['schema_version']!r}")
        if row["status"] != "success":
            raise ValueError(f"{row['comparison_id']} {row['variant_id']} did not succeed")
        identity = (str(row["comparison_id"]), str(row["variant_id"]))
        if identity in indexed:
            raise ValueError(f"duplicate formal row: {identity}")
        indexed[identity] = row

    comparison_ids = sorted({identity[0] for identity in indexed})
    if not comparison_ids:
        raise ValueError("no matching comparison rows")
    missing = [
        comparison_id
        for comparison_id in comparison_ids
        if (comparison_id, baseline) not in indexed or (comparison_id, candidate) not in indexed
    ]
    if missing:
        raise ValueError(f"missing baseline/candidate pairs: {missing}")

    per_case = []
    ratios = []
    weights = []
    paired_samples = []
    for comparison_id in comparison_ids:
        baseline_row = indexed[(comparison_id, baseline)]
        candidate_row = indexed[(comparison_id, candidate)]
        if baseline_row["case_id"] != candidate_row["case_id"]:
            raise ValueError(f"{comparison_id}: paired rows have different case_id values")
        pair_order = baseline_row["pair_order"]
        if not isinstance(pair_order, list) or not pair_order:
            raise ValueError(f"{comparison_id}: pair_order must be a non-empty list")
        if pair_order != candidate_row["pair_order"]:
            raise ValueError(f"{comparison_id}: paired rows record different pair_order values")
        baseline_latency = _latency(baseline_row)
        candidate_latency = _latency(candidate_row)
        baseline_samples = [float(value) for value in baseline_row["latency_samples_ms"]]
        candidate_samples = [float(value) for value in candidate_row["latency_samples_ms"]]
        if len(baseline_samples) != len(candidate_samples):
            raise ValueError(f"{comparison_id}: paired variants require the same sample count")
        baseline_weight = float(baseline_row.get("weight", 1.0))
        candidate_weight = float(candidate_row.get("weight", 1.0))
        if baseline_weight != candidate_weight or baseline_weight <= 0:
            raise ValueError(f"{comparison_id}: paired rows require the same positive weight")
        ratio = candidate_latency / baseline_latency
        ratios.append(ratio)
        weights.append(baseline_weight)
        paired_samples.append((baseline_samples, candidate_samples))
        per_case.append(
            {
                "comparison_id": comparison_id,
                "case_id": baseline_row["case_id"],
                "weight": baseline_weight,
                "baseline_latency_ms": baseline_latency,
                "candidate_latency_ms": candidate_latency,
                "latency_ratio": ratio,
            }
        )

    rng = random.Random(seed)
    bootstrap = []
    for _ in range(bootstrap_resamples):
        resampled_ratios = []
        for baseline_samples, candidate_samples in paired_samples:
            indices = [rng.randrange(len(baseline_samples)) for _ in baseline_samples]
            baseline_latency = statistics.median(baseline_samples[index] for index in indices)
            candidate_latency = statistics.median(candidate_samples[index] for index in indices)
            resampled_ratios.append(candidate_latency / baseline_latency)
        bootstrap.append(_weighted_geomean(resampled_ratios, weights))
    bootstrap.sort()
    alpha = 1 - confidence_level
    geomean_ratio = _weighted_geomean(ratios, weights)
    lower = _percentile(bootstrap, alpha / 2)
    upper = _percentile(bootstrap, 1 - alpha / 2)
    worst = max(ratios)
    passed = upper <= 1 - minimum_improvement and worst <= 1 + maximum_case_regression
    return {
        "schema_version": "1.0",
        "baseline_variant": baseline,
        "candidate_variant": candidate,
        "case_count": len(ratios),
        "weighted_geomean_latency_ratio": geomean_ratio,
        "confidence_level": confidence_level,
        "confidence_interval": [lower, upper],
        "worst_case_latency_ratio": worst,
        "minimum_improvement": minimum_improvement,
        "maximum_case_regression": maximum_case_regression,
        "bootstrap_resamples": bootstrap_resamples,
        "seed": seed,
        "passed": passed,
        "per_case": per_case,
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-improvement", type=float, required=True)
    parser.add_argument("--maximum-case-regression", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.confidence_level < 1:
        parser.error("confidence level must be between 0 and 1")
    if args.bootstrap_resamples < 100:
        parser.error("bootstrap-resamples must be at least 100")
    if min(args.minimum_improvement, args.maximum_case_regression) < 0:
        parser.error("performance thresholds must be non-negative")
    try:
        result = compare(
            _load_rows(args.input),
            args.baseline,
            args.candidate,
            args.confidence_level,
            args.bootstrap_resamples,
            args.seed,
            args.minimum_improvement,
            args.maximum_case_regression,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if args.output:
        _atomic_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
