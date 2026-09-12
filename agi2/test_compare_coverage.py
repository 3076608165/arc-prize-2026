"""Regression tests for separating real candidate coverage from filler accuracy."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from arclib.loader import Pair, Task
from coverage_audit import audit
from compare_coverage import compare, load_run


class ComparisonTests(unittest.TestCase):
    def test_coverage_and_quality_are_separate(self):
        tasks = [Task(key, [], [Pair([[0]], [[1]])]) for key in ("a", "b", "c")]
        with tempfile.TemporaryDirectory() as temporary:
            directories = [Path(temporary) / arm for arm in ("baseline", "candidate")]
            for directory, covered, correct in zip(directories, ({"a"}, {"a", "b"}), ({"c"}, {"a", "c"})):
                directory.mkdir()
                with patch.dict(os.environ, ARC_COVERAGE_DIR=str(directory / "coverage_audit")):
                    for task in tasks:
                        audit(-1, task.task_id, "queued", expected_outputs=1)
                        if task.task_id in covered:
                            audit(0, task.task_id, "task_started")
                            audit(0, task.task_id, "candidates_written", output_id=task.task_id + "_0", count=1)
                submission = {task.task_id: [{"attempt_1": [[1 if task.task_id in correct else 0]],
                                             "attempt_2": [[0]]}] for task in tasks}
                (directory / "submission.json").write_text(json.dumps(submission))
            result = compare(*directories, tasks)
            self.assertEqual(result["coverage_difference_outputs"], 1)
            self.assertEqual(result["candidate_only_covered"], ["b_0"])
            self.assertEqual(result["common_candidate_outputs"]["gained"], ["a_0"])
            self.assertEqual(result["common_candidate_outputs"]["n"], 1)
            # Correct filler c counts toward the final metric, never coverage.
            self.assertEqual(result["candidate"]["all_public_outputs"]["solved_outputs"], 2)
            self.assertEqual(result["candidate"]["coverage"]["outputs_with_candidates"], 2)

    def test_missing_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "absent or invalid"):
                load_run(Path(temporary), [])


if __name__ == "__main__":
    unittest.main()
