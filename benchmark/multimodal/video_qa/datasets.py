from __future__ import annotations

import ast
import json
import string
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.multimodal.common.video_io import VIDEO_EXTENSIONS, iter_video_files


CHOICE_LETTERS = "ABCDEFGH"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
EXPECTED_ROWS = {
    "mvbench": 4000,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "questions", "examples", "annotations"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [payload]
    raise ValueError(f"Unsupported annotation payload type: {type(payload)!r}")


def _list_from_value(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return [part.strip() for part in stripped.split("|") if part.strip()]
        return _list_from_value(parsed)
    return [value]


def normalize_labeled_options(options: Any) -> list[str]:
    values = [str(item).strip() for item in _list_from_value(options)]
    if not 1 <= len(values) <= len(CHOICE_LETTERS):
        raise ValueError(
            f"Expected 1-{len(CHOICE_LETTERS)} options for multiple-choice scoring, got {len(values)}: {values!r}"
        )
    labeled = []
    for idx, value in enumerate(values):
        letter = CHOICE_LETTERS[idx]
        if value[:1].upper() == letter and len(value) > 1 and value[1] in {".", ")", ":", " "}:
            labeled.append(value)
        else:
            labeled.append(f"{letter}. {value}")
    return labeled


def answer_to_letter(answer: Any, options: list[str]) -> str:
    if answer is None:
        raise ValueError("Missing answer.")
    if isinstance(answer, bool):
        raise ValueError(f"Boolean answer is not a valid multiple-choice answer: {answer!r}")
    if isinstance(answer, int):
        if 0 <= answer < len(options):
            return CHOICE_LETTERS[answer]
        if 1 <= answer <= len(options):
            return CHOICE_LETTERS[answer - 1]
    raw = str(answer).strip()
    if not raw:
        raise ValueError("Empty answer.")
    first = raw[:1].upper()
    if first in CHOICE_LETTERS[: len(options)]:
        return first
    raw_norm = _normalize_text(raw)
    for idx, option in enumerate(options):
        option_text = option.split(".", 1)[-1].strip() if "." in option[:3] else option
        if raw_norm == _normalize_text(option_text) or raw_norm == _normalize_text(option):
            return CHOICE_LETTERS[idx]
    raise ValueError(f"Cannot map answer={answer!r} to options={options!r}")


def _normalize_text(text: str) -> str:
    table = str.maketrans("", "", string.punctuation)
    return " ".join(str(text).lower().translate(table).split())


def build_video_index(video_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in iter_video_files(video_dir):
        stem = path.stem
        index.setdefault(stem, path)
        index.setdefault(path.name, path)
        rel = str(path.relative_to(video_dir))
        index.setdefault(rel, path)
    for path in video_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        frame_dir = path.parent
        index.setdefault(frame_dir.name, frame_dir)
        rel = str(frame_dir.relative_to(video_dir))
        index.setdefault(rel, frame_dir)
    return index


def _is_media_candidate(path: Path) -> bool:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
        return True
    if path.is_dir():
        return any(child.suffix.lower() in IMAGE_EXTENSIONS for child in path.iterdir() if child.is_file())
    return False


def resolve_video_path(raw_value: Any, dataset_dir: Path, video_dir: Path, video_index: dict[str, Path]) -> Path:
    raw = str(raw_value or "").strip()
    if not raw:
        raise FileNotFoundError("Row has no video path/id field.")
    raw_path = Path(raw)
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.extend(
        [
            dataset_dir / raw_path,
            video_dir / raw_path,
            video_dir / raw_path.name,
        ]
    )
    for candidate in candidates:
        if candidate.exists() and _is_media_candidate(candidate):
            return candidate
    keys = [raw, raw_path.name, raw_path.stem]
    for key in keys:
        if key in video_index:
            return video_index[key]
    raise FileNotFoundError(f"Cannot resolve video={raw!r}; tried keys={keys!r}")


def _find_first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _make_row(
    *,
    benchmark: str,
    raw: dict[str, Any],
    annotation_name: str,
    row_idx: int,
    dataset_dir: Path,
    video_dir: Path,
    video_index: dict[str, Path],
) -> dict[str, Any]:
    question = str(_find_first(raw, ("question", "Q", "query", "prompt", "question_text"), "")).strip()
    if not question:
        raise ValueError("Missing question.")

    options = _find_first(raw, ("options", "candidates", "candidate", "choices", "answers"))
    if options is None:
        letter_options = []
        for letter in CHOICE_LETTERS:
            for key in (letter, f"option_{letter}", f"option{letter}", f"answer_{letter}"):
                if key in raw:
                    letter_options.append(raw[key])
                    break
        options = letter_options
    labeled_options = normalize_labeled_options(options)
    answer = answer_to_letter(_find_first(raw, ("answer", "label", "correct_answer", "gt", "ground_truth")), labeled_options)
    video_value = _find_first(raw, ("video", "video_path", "video_id", "videoID", "video_name", "vid", "filename", "file"))
    video_path = resolve_video_path(video_value, dataset_dir, video_dir, video_index)
    qid = str(_find_first(raw, ("question_id", "qid", "id", "sample_id"), f"{annotation_name}:{row_idx}"))
    task_type = str(_find_first(raw, ("task_type", "task", "category", "sub_task", "question_type"), annotation_name))
    time_stamp = str(_find_first(raw, ("time_stamp", "timestamp", "time", "duration"), "full"))
    context = str(_find_first(raw, ("context", "subtitle", "subtitles"), "") or "")
    return {
        "benchmark": benchmark,
        "task": annotation_name,
        "task_type": task_type,
        "question_id": qid,
        "sample_id": row_idx,
        "question": question,
        "time_stamp": time_stamp,
        "timestamp_seconds": 1.0e9,
        "answer": answer,
        "options": labeled_options,
        "context": context,
        "video_path": str(video_path),
    }


def _load_json_benchmark(
    *,
    benchmark: str,
    dataset_dir: Path,
    annotation_dir: Path,
    video_dir: Path,
    annotation_glob: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Missing {benchmark} annotation directory: {annotation_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing {benchmark} video directory: {video_dir}")
    video_index = build_video_index(video_dir)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    annotation_files = sorted(annotation_dir.glob(annotation_glob))
    if not annotation_files:
        raise FileNotFoundError(f"No {benchmark} annotation files matched {annotation_dir / annotation_glob}")
    for annotation_path in annotation_files:
        records = _iter_records(_read_json(annotation_path))
        annotation_name = annotation_path.stem
        for idx, raw in enumerate(records):
            try:
                rows.append(
                    _make_row(
                        benchmark=benchmark,
                        raw=raw,
                        annotation_name=annotation_name,
                        row_idx=len(rows),
                        dataset_dir=dataset_dir,
                        video_dir=video_dir,
                        video_index=video_index,
                    )
                )
            except Exception as exc:
                skipped.append(
                    {
                        "annotation": str(annotation_path),
                        "row_index": idx,
                        "reason": str(exc),
                    }
                )
    skipped_by_reason: dict[str, int] = {}
    for item in skipped:
        reason = str(item["reason"])
        if "Cannot resolve video=" in reason:
            key = "missing_video_or_frame_dir"
        elif "options for multiple-choice scoring" in reason:
            key = "invalid_option_count"
        elif "Cannot map answer=" in reason:
            key = "invalid_answer"
        else:
            key = "parse_error"
        skipped_by_reason[key] = skipped_by_reason.get(key, 0) + 1

    info = {
        "benchmark": benchmark,
        "dataset_dir": str(dataset_dir),
        "annotation_dir": str(annotation_dir),
        "video_dir": str(video_dir),
        "annotation_files": [str(path) for path in annotation_files],
        "indexed_video_count": len(video_index),
        "selected_rows_before_slice": len(rows),
        "skipped_rows": len(skipped),
        "skipped_by_reason": skipped_by_reason,
        "skipped_examples": skipped[:20],
    }
    if benchmark in EXPECTED_ROWS:
        info["expected_row_count"] = EXPECTED_ROWS[benchmark]
        info["matches_expected_row_count"] = len(rows) == EXPECTED_ROWS[benchmark]
    return rows, info


def _slice_rows(rows: list[dict[str, Any]], sample_start: int, num_samples: int) -> list[dict[str, Any]]:
    selected = rows[max(0, int(sample_start)) :]
    if num_samples >= 0:
        selected = selected[:num_samples]
    return selected


def load_mvbench_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else dataset_dir / "json"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "video"
    rows, info = _load_json_benchmark(
        benchmark="mvbench",
        dataset_dir=dataset_dir,
        annotation_dir=annotation_dir,
        video_dir=video_dir,
        annotation_glob="*.json",
    )
    rows = _slice_rows(rows, args.sample_start, args.num_samples)
    info["evaluated_sample_count"] = len(rows)
    return rows, info


def load_mlvu_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    annotation_dir = Path(args.annotation_dir) if args.annotation_dir else dataset_dir / "MLVU" / "json"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "MLVU" / "video"
    rows, info = _load_json_benchmark(
        benchmark="mlvu",
        dataset_dir=dataset_dir,
        annotation_dir=annotation_dir,
        video_dir=video_dir,
        annotation_glob="*.json",
    )
    rows = _slice_rows(rows, args.sample_start, args.num_samples)
    info["evaluated_sample_count"] = len(rows)
    return rows, info


def _read_table(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    if path.suffix in {".json", ".jsonl"}:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return _iter_records(_read_json(path))
    raise ValueError(f"Unsupported annotation table type: {path}")


def load_longvideobench_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    annotation_path = Path(args.annotation_path) if args.annotation_path else dataset_dir / "lvb_val.json"
    video_dir = Path(args.video_dir) if args.video_dir else dataset_dir / "videos"
    if not annotation_path.exists():
        parquet = dataset_dir / "validation-00000-of-00001.parquet"
        annotation_path = parquet if parquet.exists() else annotation_path
    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing LongVideoBench annotation file: {annotation_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing LongVideoBench video directory: {video_dir}")
    video_index = build_video_index(video_dir)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for idx, raw in enumerate(_read_table(annotation_path)):
        try:
            rows.append(
                _make_row(
                    benchmark="longvideobench",
                    raw=raw,
                    annotation_name=annotation_path.stem,
                    row_idx=len(rows),
                    dataset_dir=dataset_dir,
                    video_dir=video_dir,
                    video_index=video_index,
                )
            )
        except Exception as exc:
            skipped.append({"annotation": str(annotation_path), "row_index": idx, "reason": str(exc)})
    rows_before_slice = len(rows)
    rows = _slice_rows(rows, args.sample_start, args.num_samples)
    return rows, {
        "benchmark": "longvideobench",
        "dataset_dir": str(dataset_dir),
        "annotation_path": str(annotation_path),
        "video_dir": str(video_dir),
        "indexed_video_count": len(video_index),
        "selected_rows_before_slice": rows_before_slice,
        "evaluated_sample_count": len(rows),
        "skipped_rows": len(skipped),
        "skipped_examples": skipped[:20],
    }


def load_videomme_unified_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from benchmark.multimodal.video_qa.videomme import load_videomme_rows

    rows, info = load_videomme_rows(args, resolve_videos=not getattr(args, "dry_run_metadata", False))
    for row in rows:
        row["benchmark"] = "videomme"
    info["benchmark"] = "videomme"
    info["evaluated_sample_count"] = len(rows)
    return rows, info


def load_video_qa_rows(args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    benchmark = str(args.benchmark).lower()
    if benchmark == "mvbench":
        return load_mvbench_rows(args)
    if benchmark == "mlvu":
        return load_mlvu_rows(args)
    if benchmark == "longvideobench":
        return load_longvideobench_rows(args)
    if benchmark == "videomme":
        return load_videomme_unified_rows(args)
    raise ValueError(f"Unsupported video benchmark: {args.benchmark!r}")
