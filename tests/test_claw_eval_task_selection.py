import json
import tempfile
import unittest
from pathlib import Path

from benchmark.claw_eval.select_tasks import TaskSelectionError
from benchmark.claw_eval.select_tasks import select_text_only_tasks


def _write_task(
    tasks_dir: Path,
    task_id: str,
    *,
    category: str = "workflow",
    files: list[str] | None = None,
    tags: list[str] | None = None,
) -> None:
    task_dir = tasks_dir / task_id
    task_dir.mkdir(parents=True)
    lines = [
        f"task_id: {task_id}",
        f"task_name: {task_id}",
        f"category: {category}",
        "tags: [" + ", ".join(tags or ["general"]) + "]",
        "prompt:",
        "  text: test",
        "sandbox_files:",
    ]
    lines.extend(f"  - {value}" for value in files or [])
    (task_dir / "task.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


class ClawEvalTaskSelectionTest(unittest.TestCase):
    def test_text_only_policy_records_visual_tasks_as_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "claw-eval"
            tasks_dir = source_root / "tasks"
            (source_root / "mock_services").mkdir(parents=True)
            _write_task(tasks_dir, "T001", files=["fixtures/data.json"])
            _write_task(tasks_dir, "T002", files=["fixtures/report.pdf"])
            _write_task(tasks_dir, "T003", category="multimodal")
            _write_task(tasks_dir, "T004", tags=["other"])
            _write_task(tasks_dir, "T005", tags=["general", "multimodal"])
            _write_task(tasks_dir, "T006", files=["fixtures/diagram.svg"])
            output_root = root / "selection"
            summary_path = root / "selection.json"
            skipped_path = root / "skipped.jsonl"

            summary = select_text_only_tasks(
                source_tasks_dir=tasks_dir,
                output_root=output_root,
                tag="general",
                summary_path=summary_path,
                skipped_results_path=skipped_path,
            )

            self.assertEqual(summary["selected_count"], 1)
            self.assertEqual(summary["skipped_count"], 4)
            self.assertTrue((output_root / "tasks" / "T001").is_symlink())
            self.assertTrue((output_root / "mock_services").is_symlink())
            rows = [json.loads(line) for line in skipped_path.read_text().splitlines()]
            self.assertEqual(
                {row["task_id"] for row in rows},
                {"T002", "T003", "T005", "T006"},
            )
            self.assertTrue(all(row["status"] == "skipped_by_policy" for row in rows))

    def test_selection_refuses_unowned_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_dir = root / "claw-eval" / "tasks"
            _write_task(tasks_dir, "T001")
            output_root = root / "selection"
            output_root.mkdir()
            (output_root / "user-file").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(TaskSelectionError, "Refusing to reuse"):
                select_text_only_tasks(
                    source_tasks_dir=tasks_dir,
                    output_root=output_root,
                    tag="general",
                    summary_path=root / "summary.json",
                    skipped_results_path=root / "skipped.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
