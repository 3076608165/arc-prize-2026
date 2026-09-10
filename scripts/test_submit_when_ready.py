import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import submit_when_ready as monitor


class SubmitTests(unittest.TestCase):
    def args(self, receipt):
        return ["monitor", "--kernel", "owner/kernel", "--version", "10",
                "--competition", "competition", "--file", "submission.json",
                "--message", "unique-experiment", "--receipt", str(receipt)]

    def test_submits_pinned_version_only_once(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            with patch.object(sys, "argv", self.args(receipt)), patch.object(
                    monitor, "kaggle", side_effect=["KernelWorkerStatus.COMPLETE",
                                                   "ref,description\n", "success",
                                                   "ref,description\n123,unique-experiment\n"]) as api:
                monitor.main()
                self.assertEqual(api.call_args_list[2].args,
                                 ("competitions", "submit", "competition", "-k", "owner/kernel",
                                  "-v", "10", "-f", "submission.json", "-m", "unique-experiment"))
                self.assertEqual(json.loads(receipt.read_text())["status"], "submitted")
                api.reset_mock()
                with self.assertRaises(SystemExit):
                    monitor.main()
                api.assert_not_called()

    def test_uncertain_response_blocks_automatic_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            with patch.object(sys, "argv", self.args(receipt)), patch.object(
                    monitor, "kaggle", side_effect=["KernelWorkerStatus.COMPLETE",
                                                   "ref,description\n", RuntimeError("connection lost")]):
                with self.assertRaises(RuntimeError):
                    monitor.main()
                self.assertEqual(json.loads(receipt.read_text())["status"], "submission_uncertain")


if __name__ == "__main__":
    unittest.main()
