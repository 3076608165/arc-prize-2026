import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import submit_agi3_after_evaluation as worker


class SubmissionTests(unittest.TestCase):
    def fixture(self, root):
        trial = root / "trial"
        trial.mkdir()
        source = "class MyAgent: pass\n"
        protocol = {"environment": {"test": "hash"}, "arms": {}}
        for arm in ("stable", "coverage"):
            (trial / f"{arm}.py").write_bytes(source.replace("\n", "\r\n").encode())
            sha = hashlib.sha256((trial / f"{arm}.py").read_bytes()).hexdigest()
            protocol["arms"][arm] = {"source": f"{arm}.py", "sha256": sha}
            bundle = root / f"agi3/out/{arm}-20260911"
            bundle.mkdir(parents=True)
            (bundle / "submission.ipynb").write_text(json.dumps({"cells": [{"cell_type": "code", "source": "%%writefile /tmp/my_agent.py\n" + source}]}))
            (bundle / "kernel-metadata.json").write_text(json.dumps({"id": "uts3125025522741/arc-agi-3-sweep-agent", "is_private": True}))
        (trial / "protocol.json").write_text(json.dumps(protocol))
        (trial / "results.json").write_text("[]")
        return trial

    def test_complete_gate_selects_candidate_or_fallback_once(self):
        for status, expected in (("eligible_for_online_experiment", "coverage"),
                                 ("inconclusive_do_not_promote", "stable")):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trial = self.fixture(root)
                with patch.multiple(worker, ROOT=root, TRIAL=trial, OUT=root / "output"), \
                     patch.object(worker, "analyze", return_value={"status": status}), \
                     patch.object(worker, "environment_fingerprint", return_value={"test": "hash"}), \
                     patch.object(worker, "kaggle", return_value="Kernel version 4 successfully pushed") as push, \
                     patch.object(worker.subprocess, "run") as submit:
                    worker.main()
                    self.assertEqual(push.call_count, 1)
                    self.assertIn(expected + "-20260911", push.call_args.args[-1])
                    self.assertEqual(submit.call_count, 1)
                    args = submit.call_args.args[0]
                    self.assertEqual(args[args.index("--version") + 1], "4")
                    with self.assertRaises(SystemExit):
                        worker.main()
                    self.assertEqual(push.call_count, 1)

    def test_incomplete_experiment_does_not_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = self.fixture(root)
            with patch.multiple(worker, ROOT=root, TRIAL=trial, OUT=root / "output"), \
                 patch.object(worker, "analyze", return_value={"status": "collecting"}), \
                 patch.object(worker.time, "sleep", side_effect=RuntimeError("stop test wait")), \
                 patch.object(worker, "kaggle") as push:
                with self.assertRaises(RuntimeError):
                    worker.main()
                push.assert_not_called()


if __name__ == "__main__":
    unittest.main()
