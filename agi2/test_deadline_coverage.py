import json
from pathlib import Path
import tempfile
import unittest
from deadline_coverage import summarize_deadline


class DeadlineTests(unittest.TestCase):
    def evaluate(self, rows, expected):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "rank-0.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            return summarize_deadline(Path(directory), expected)

    def test_late_rewrite_is_not_new_coverage(self):
        rows = [{"rank": 0, "event": "worker_started", "time": 0, "deadline": 10}]
        for output, when in (("a_0", 9), ("a_0", 11), ("b_0", 12)):
            rows.append(dict(rank=0, event="candidates_written", output_id=output, time=when, count=1))
        result = self.evaluate(rows, {"a_0", "b_0"})
        self.assertEqual(result["outputs_confirmed_by_deadline"], ["a_0"])
        self.assertEqual(result["outputs_first_observed_after_deadline"], ["b_0"])

    def test_missing_deadline_keeps_coverage_unknown(self):
        result = self.evaluate([dict(rank=0, event="candidates_written", output_id="a_0", time=1, count=1)], {"a_0"})
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["outputs_with_unknown_timing"], ["a_0"])

    def test_missing_raw_output_is_not_silently_counted(self):
        result = self.evaluate([dict(rank=0, event="worker_started", time=0, deadline=10)], {"a_0"})
        self.assertEqual(result["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
