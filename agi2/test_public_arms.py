"""Public arms are lifted from other people's notebooks, so the lifts get verified.

Each of these can break silently in a way that costs a day's slot:

  - **Leg C** (swarm-arc2-legc-measure): the 45k-char module arrives truncated or
    half-copied (the notebook file is one line of JSON), the four cells land in
    an order that makes the merge a no-op or a crash, or the coder model is not
    attached -- in which case the pre-pass writes empty results, which looks
    exactly like "Leg C does not help";
  - **original numerics** (swarm-arc2-numerics-control): the perfpatch block is
    matched only partially, so the notebook keeps the shortcut and the arm
    measures nothing.

The model attach is checked at build level, on the metadata main() writes.
"""

import ast
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from repro_nvarc import (LEGC_MODEL, LEGC_REF, add_legc, cell_text,
                         fetch_upstream, use_original_numerics)


def ref_module_text() -> str:
    nb = json.loads(LEGC_REF.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            txt = cell_text(cell)
            if txt.startswith("%%writefile arc_induction_v2.py"):
                return txt
    raise AssertionError("reference module cell not found")


def upstream_notebook() -> dict:
    return json.loads(fetch_upstream().read_text(encoding="utf-8"))


def texts(nb: dict) -> list[str]:
    return [cell_text(c) if c.get("cell_type") == "code" else "" for c in nb["cells"]]


class LegCTests(unittest.TestCase):

    def test_module_is_copied_byte_for_byte(self):
        nb = upstream_notebook()
        self.assertTrue(add_legc(nb))
        module = next(t for t in texts(nb) if t.startswith("%%writefile arc_induction_v2.py"))
        ref = ref_module_text()
        self.assertEqual(hashlib.sha256(module.encode()).hexdigest(),
                         hashlib.sha256(ref.encode()).hexdigest())
        self.assertGreater(len(module), 40000)
        body = module.split("\n", 1)[1]
        ast.parse(body)  # the DSL runtime and sandbox are syntactically valid

    def test_cells_land_in_the_only_order_that_works(self):
        nb = upstream_notebook()
        self.assertTrue(add_legc(nb))
        t = texts(nb)
        write = max(i for i, x in enumerate(t)
                    if x.startswith("%%writefile ") and "arc_induction_v2" not in x)
        module = next(i for i, x in enumerate(t) if x.startswith("%%writefile arc_induction_v2.py"))
        selftest = next(i for i, x in enumerate(t) if "arc_induction_v2.py --selftest" in x)
        launch = next(i for i, x in enumerate(t) if "Leg C launch" in x)
        run = next(i for i, x in enumerate(t) if "python starter.py" in x)
        sub = next(i for i, x in enumerate(t) if "data.get_submission(" in x)
        merge = next(i for i, x in enumerate(t) if "Leg C merge" in x)
        self.assertEqual(module, write + 1)     # after every other writefile
        self.assertEqual(selftest, module + 1)  # sandbox proof before any GPU work
        self.assertEqual(launch, run - 1)       # pre-pass before the base pipeline
        self.assertEqual(merge, sub + 1)        # merge after submission.json exists
        self.assertLess(module, launch)
        merge_txt = t[merge]
        self.assertIn("attempt_1", merge_txt)
        self.assertIn("induction_results.json", merge_txt)

    def test_budget_constants_are_ours_not_theirs(self):
        nb = upstream_notebook()
        self.assertTrue(add_legc(nb, budget_h=0.5, reserve_h=10.5))
        launch = next(t for t in texts(nb) if "Leg C launch" in t)
        self.assertIn("LEGC_BUDGET_H  = 0.5", launch)
        self.assertIn("BASE_RESERVE_H = 10.5", launch)
        self.assertIn("global_end_time", launch)  # needs our budget cell

    def test_refuses_a_notebook_without_the_anchors(self):
        self.assertFalse(add_legc({"cells": [
            {"cell_type": "code", "source": ["print('nothing to patch')\n"]}]}))

    def test_build_attaches_the_coder_model(self):
        """The pre-pass silently no-ops when the model is absent, so the metadata
        entry is part of the feature, not packaging."""
        import repro_nvarc
        with tempfile.TemporaryDirectory() as tmp:
            argv = sys.argv
            sys.argv = ["repro_nvarc.py", "--legc", "--legc-budget-h", "1.0",
                        "--out", tmp, "--kernel-id", "someone/nb", "--title", "nb"]
            try:
                self.assertEqual(repro_nvarc.main(), 0)
            finally:
                sys.argv = argv
            meta = json.loads((Path(tmp) / "kernel-metadata.json").read_text(encoding="utf-8"))
            exp = json.loads((Path(tmp) / "experiment.json").read_text(encoding="utf-8"))
        self.assertIn(LEGC_MODEL, meta["model_sources"])
        self.assertGreaterEqual(len(meta["model_sources"]), 2)
        self.assertTrue(exp["legc"])
        self.assertEqual(exp["legc_budget_h"], 1.0)


class OriginalNumericsTests(unittest.TestCase):

    def test_the_shortcut_is_gone_and_the_baseline_arithmetic_is_back(self):
        nb = upstream_notebook()
        self.assertTrue(use_original_numerics(nb))
        solver = next(t for t in texts(nb) if "def turbo_dfs(" in t)
        self.assertIn("logits.float().cpu().log_softmax(-1)", solver)
        self.assertNotIn("torch.logsumexp(logits_f, dim=-1, keepdim=True)", solver)
        self.assertIn("for t in ARC_TOKENS:\n            score = nll[i, t].item()", solver)
        self.assertNotIn("enumerate(ARC_TOKENS)", solver)
        ast.parse(solver.split("\n", 1)[1])

    def test_it_flips_exactly_one_file(self):
        """Only arc_solver.py holds turbo_dfs; a second match would mean we patched
        a stale copy and the notebook would keep the shortcut."""
        nb = upstream_notebook()
        before = texts(nb)
        self.assertTrue(use_original_numerics(nb))
        changed = [i for i, (a, b) in enumerate(zip(before, texts(nb))) if a != b]
        self.assertEqual(len(changed), 1)
        self.assertIn("turbo_dfs", before[changed[0]])

    def test_refuses_a_notebook_without_the_block(self):
        self.assertFalse(use_original_numerics({"cells": [
            {"cell_type": "code", "source": ["def turbo_dfs(): pass\n"]}]}))


if __name__ == "__main__":
    unittest.main()
