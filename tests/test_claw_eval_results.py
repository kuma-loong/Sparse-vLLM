import json
import tempfile
import unittest
from pathlib import Path

from benchmark.claw_eval.validate_results import (
    ClawResultError,
    validate_changed_results,
    write_snapshot,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _trial(*, passed: bool = False, error: str | None = None) -> dict:
    trial = {
        "passed": passed,
        "task_score": 1.0 if passed else 0.2,
        "tokens": 12,
    }
    if error is not None:
        trial["error"] = error
    return trial


class ClawEvalResultValidationTest(unittest.TestCase):
    def test_failed_score_is_a_successful_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            snapshot = root / "before.json"
            per_sample = root / "per_sample_results.jsonl"
            final_summary = root / "final_summary.json"
            write_snapshot(trace_dir, snapshot)
            batch_dir = trace_dir / "model_run"
            _write_json(
                batch_dir / "batch_results.json",
                [{"task_id": "T100", "error": None, "trials": [_trial()]}],
            )
            _write_json(
                batch_dir / "batch_summary.json",
                {"tasks": 1, "trials_per_task": 1, "errored": 0, "avg_score": 0.2},
            )

            summary = validate_changed_results(
                trace_dir=trace_dir,
                snapshot_path=snapshot,
                per_sample_path=per_sample,
                final_summary_path=final_summary,
            )

            row = json.loads(per_sample.read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "success")
            self.assertFalse(row["resolved"])
            self.assertEqual(summary["status_counts"], {"success": 1})

    def test_task_error_is_written_and_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            snapshot = root / "before.json"
            per_sample = root / "per_sample_results.jsonl"
            final_summary = root / "final_summary.json"
            write_snapshot(trace_dir, snapshot)
            batch_dir = trace_dir / "model_run"
            _write_json(
                batch_dir / "batch_results.json",
                [{"task_id": "T100", "error": "model unavailable", "trials": []}],
            )
            _write_json(
                batch_dir / "batch_summary.json",
                {"tasks": 1, "trials_per_task": 1, "errored": 1, "avg_score": 0.0},
            )

            with self.assertRaisesRegex(ClawResultError, "1 task"):
                validate_changed_results(
                    trace_dir=trace_dir,
                    snapshot_path=snapshot,
                    per_sample_path=per_sample,
                    final_summary_path=final_summary,
                )

            row = json.loads(per_sample.read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "metric_failed")
            self.assertIn("model unavailable", row["error"])

    def test_unchanged_results_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            batch_dir = trace_dir / "model_run"
            _write_json(batch_dir / "batch_results.json", [])
            _write_json(
                batch_dir / "batch_summary.json",
                {"tasks": 0, "trials_per_task": 1, "errored": 0},
            )
            snapshot = root / "before.json"
            write_snapshot(trace_dir, snapshot)

            with self.assertRaisesRegex(ClawResultError, "changed batch result"):
                validate_changed_results(
                    trace_dir=trace_dir,
                    snapshot_path=snapshot,
                    per_sample_path=root / "per_sample.jsonl",
                    final_summary_path=root / "summary.json",
                )

    def test_policy_skips_are_preserved_without_failing_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_dir = root / "traces"
            snapshot = root / "before.json"
            per_sample = root / "per_sample_results.jsonl"
            final_summary = root / "final_summary.json"
            skipped = root / "skipped.jsonl"
            skipped.write_text(
                json.dumps(
                    {
                        "task_id": "T200",
                        "status": "skipped_by_policy",
                        "resolved": None,
                        "score": None,
                        "trials": 0,
                        "error": None,
                        "skip_reason": "visual_files=fixture.pdf",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            write_snapshot(trace_dir, snapshot)
            batch_dir = trace_dir / "model_run"
            _write_json(
                batch_dir / "batch_results.json",
                [{"task_id": "T100", "error": None, "trials": [_trial(passed=True)]}],
            )
            _write_json(
                batch_dir / "batch_summary.json",
                {"tasks": 1, "trials_per_task": 1, "errored": 0, "avg_score": 1.0},
            )

            summary = validate_changed_results(
                trace_dir=trace_dir,
                snapshot_path=snapshot,
                per_sample_path=per_sample,
                final_summary_path=final_summary,
                skipped_results_path=skipped,
            )

            rows = [json.loads(line) for line in per_sample.read_text().splitlines()]
            self.assertEqual(summary["tasks"], 1)
            self.assertEqual(summary["skipped_tasks"], 1)
            self.assertEqual(summary["total_scope_tasks"], 2)
            self.assertEqual(
                summary["status_counts"],
                {"skipped_by_policy": 1, "success": 1},
            )
            self.assertEqual({row["task_id"] for row in rows}, {"T100", "T200"})


if __name__ == "__main__":
    unittest.main()
