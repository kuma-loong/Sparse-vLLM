#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


COGNITION_CATEGORIES = {
    "code_reasoning",
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute official-style MME acc+acc_plus scores.")
    parser.add_argument("--per_sample_results", required=True, help="Path to *_per_sample_results.jsonl from MME eval.")
    parser.add_argument("--output_json", required=True, help="Path to write the MME score JSON.")
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing per-sample result file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("benchmark") != "mme":
                raise ValueError(f"Line {line_no} is not an MME record: benchmark={record.get('benchmark')!r}")
            category = record.get("category")
            question_id = record.get("question_id")
            status = record.get("status")
            if not category:
                raise ValueError(f"Line {line_no} is missing category.")
            if not question_id:
                raise ValueError(f"Line {line_no} is missing question_id.")
            if status != "success":
                raise ValueError(f"Line {line_no} has non-success status={status!r}; fix parsing before scoring MME.")
            if not isinstance(record.get("correct"), bool):
                raise ValueError(f"Line {line_no} has non-bool correct={record.get('correct')!r}.")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def score_category(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_image[str(record["question_id"])].append(record)

    bad_groups = {qid: len(items) for qid, items in by_image.items() if len(items) != 2}
    if bad_groups:
        preview = dict(list(sorted(bad_groups.items()))[:5])
        raise ValueError(f"MME official score expects exactly 2 questions per image; bad groups={preview}")

    correct = sum(1 for record in records if record["correct"])
    accuracy = 100.0 * correct / len(records)
    plus_correct = sum(1 for items in by_image.values() if all(record["correct"] for record in items))
    accuracy_plus = 100.0 * plus_correct / len(by_image)
    return {
        "num_questions": len(records),
        "num_images": len(by_image),
        "correct": correct,
        "plus_correct": plus_correct,
        "accuracy": accuracy,
        "accuracy_plus": accuracy_plus,
        "score": accuracy + accuracy_plus,
    }


def main() -> None:
    args = parse_args()
    records = read_records(Path(args.per_sample_results))
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_category[str(record["category"])].append(record)

    categories = {category: score_category(items) for category, items in sorted(by_category.items())}
    cognition_score = sum(item["score"] for category, item in categories.items() if category in COGNITION_CATEGORIES)
    perception_score = sum(item["score"] for category, item in categories.items() if category not in COGNITION_CATEGORIES)
    output = {
        "metric": "MME official-style score: sum(category accuracy + accuracy_plus)",
        "num_categories": len(categories),
        "num_questions": len(records),
        "num_images": sum(item["num_images"] for item in categories.values()),
        "cognition_categories": sorted(COGNITION_CATEGORIES & set(categories)),
        "perception_categories": sorted(set(categories) - COGNITION_CATEGORIES),
        "cognition_score": cognition_score,
        "perception_score": perception_score,
        "total_score": cognition_score + perception_score,
        "categories": categories,
        "source": str(Path(args.per_sample_results)),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
