import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coverage_audit as ca
from repro_nvarc import CACHE, cell_text


class AuditTests(unittest.TestCase):
    def test_incomplete_and_filler_are_not_counted_as_covered(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, ARC_COVERAGE_DIR=directory):
            for task in ("done", "timedout", "unstarted", "crashed", "empty"):
                ca.audit(-1, task, "queued", expected_outputs=2)
            for task in ("done", "timedout", "crashed", "empty"):
                ca.audit(0, task, "task_started")
                if task == "crashed":
                    ca.audit(0, None, "worker_error", error_type="MemoryError")
                    continue
                ca.audit(0, task, "training_started")
                ca.audit(0, task, "training_completed")
                ca.audit(0, task, "batches_planned", count=2)
                ca.audit(0, task, "batch_completed")
                if task == "timedout":
                    ca.audit(0, task, "task_truncated", reason="global_deadline")
                else:
                    ca.audit(0, task, "batch_completed")
                if task == "done":
                    for output in ("done_0", "done_1"):
                        ca.audit(0, task, "candidates_written", output_id=output, count=3)
                ca.audit(0, task, "task_finished")
            # A filler submission exists but is deliberately never read by the audit.
            Path(directory, "submission.json").write_text('{"unstarted": [[[0]]]}')
            result = ca.summarize(Path(directory))
            self.assertEqual(result["counts"]["queued"], 5)
            self.assertEqual(result["counts"]["all_outputs_have_candidates"], 1)
            self.assertEqual(result["counts"]["not_started"], 1)
            self.assertEqual(result["counts"]["error"], 1)
            self.assertEqual(result["counts"]["no_valid_candidates_after_finished"], 1)
            self.assertFalse(result["tasks"]["timedout"]["decoding_completed_without_observed_cap"])

    def test_partial_event_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "rank-0.jsonl").write_text('{"task":')
            self.assertTrue(ca.summarize(Path(directory))["event_errors"])

    def test_solver_algorithm_ast_is_unchanged(self):
        paths = list(CACHE.glob("*.ipynb"))
        if not paths:
            self.skipTest("Cached upstream notebook unavailable")
        notebook = json.loads(paths[0].read_text(encoding="utf-8"))
        def solver():
            return next(cell_text(c).split("\n", 1)[1] for c in notebook["cells"]
                        if c["cell_type"] == "code" and cell_text(c).startswith("%%writefile arc_solver.py\n"))
        before = ast.parse(solver())
        ca.patch_notebook(notebook)
        after = ast.parse(solver())
        class RemoveAudit(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                return None if node.module == "coverage_audit" else node
            def visit_Expr(self, node):
                call = node.value
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "_audit":
                    return None
                return self.generic_visit(node)
            def visit_Assign(self, node):
                if any(isinstance(t, ast.Name) and t.id == "_dfs_start" for t in node.targets):
                    return None
                return node
        self.assertEqual(ast.dump(before), ast.dump(RemoveAudit().visit(after)))


if __name__ == "__main__":
    unittest.main()
