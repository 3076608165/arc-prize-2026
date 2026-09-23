"""The second pass has to be wired in three places at once, or not at all.

A half-applied second pass is the expensive failure: the notebook pushes, burns
a day's slot, and either scores nothing new (pool applied without the extra
queue: no pass B candidates exist) or loses the monotone guarantee (pass B
pulled without the attempt_1 pin: pass A's top-1 can be displaced). These tests
run against the real cached upstream notebook and compile the patched files.
"""
import ast
import json
import re
import unittest
from pathlib import Path

from repro_nvarc import (PASS2_MARKER, PASS2_STORE, add_two_pass_pooling,
                         drop_allowlist, fetch_upstream)

ROOT = Path(__file__).resolve().parents[1]


def upstream_notebook() -> dict:
    return json.loads(fetch_upstream().read_text(encoding="utf-8"))


def cell_with(nb: dict, needle: str) -> str:
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = "".join(cell["source"])
        if needle in txt:
            return txt
    raise AssertionError(f"no code cell contains {needle!r}")


def patched() -> dict:
    nb = upstream_notebook()
    assert add_two_pass_pooling(nb)
    return nb


def written_files(nb: dict) -> dict:
    """{filename: source} for every %%writefile cell, the way Kaggle runs them."""
    out = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = "".join(cell["source"])
        m = re.match(r"%%writefile (\S+)\n(.*)", txt, re.S)
        if m:
            out[m.group(1)] = m.group(2)
    return out


class TwoPassTests(unittest.TestCase):
    def test_both_passes_cover_the_same_tasks(self):
        """Pass B copies the filter lines, so a commit-mode subset stays matched.

        Without --full-eval the allowlist is still in place; a second loop that
        forgot it would queue all 120 eval tasks in a run that only mounts the
        four-task answers, i.e. spend the budget on tasks it cannot score.
        """
        nb = patched()
        starter = cell_with(nb, "_queue_keys = sorted")
        loops = starter.count("for key in _queue_keys:")
        self.assertEqual(loops, 2)
        self.assertEqual(starter.count('queue.put("0934a4d8")'), 0)  # never hardcoded
        allowlist = 'if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:'
        self.assertEqual(starter.count(allowlist), 2)
        self.assertEqual(starter.count("        queue.put(key)\n"), 2)
        self.assertLess(starter.index("_queue_keys = sorted"),
                        starter.index("for key in _queue_keys:"))
        self.assertLess(starter.index(PASS2_MARKER),
                        starter.index("for key in _queue_keys:", starter.index(PASS2_MARKER)))

    def test_after_full_eval_the_second_loop_has_no_filter(self):
        nb = upstream_notebook()
        self.assertTrue(drop_allowlist(nb))
        self.assertTrue(add_two_pass_pooling(nb))
        starter = cell_with(nb, "_queue_keys = sorted")
        self.assertEqual(starter.count("for key in _queue_keys:"), 2)
        self.assertNotIn("continue", starter.split(PASS2_MARKER)[1])
        self.assertEqual(starter.count("        queue.put(key)\n"), 2)

    def test_worker_separates_the_passes(self):
        solver = written_files(patched())["arc_solver.py"]
        self.assertIn("pass_id = 0", solver)
        self.assertIn(f'if key == "{PASS2_MARKER}":', solver)
        self.assertIn(f'dir_outputs = "{PASS2_STORE}"', solver)
        # both stores get created, and the marker never reaches the task lookup
        self.assertEqual(solver.count("os.makedirs(dir_outputs, exist_ok=True)"), 2)
        self.assertIn("seed=1 + 101 * pass_id", solver)
        self.assertIn("seed=2 + 103 * pass_id", solver)

    def test_selection_is_monotone(self):
        nb = patched()
        selection = cell_with(nb, "data.get_submission(selected)")
        self.assertIn(f'os.path.isdir("{PASS2_STORE}")', selection)
        self.assertIn("attempt_1 pinned to pass A's top-1", selection)
        # attempt_1 comes from pass A, the rest of the list from the pool
        self.assertIn("_merged[_bk] = ([_first] if _first else []) + [", selection)
        self.assertIn("hashable(c['solution']) not in _seen", selection)

    def test_patched_files_compile(self):
        for name, src in written_files(patched()).items():
            if name.endswith(".py"):
                ast.parse(src, filename=name)

    def test_is_all_or_nothing_on_a_notebook_without_the_anchors(self):
        self.assertFalse(add_two_pass_pooling({"cells": [
            {"cell_type": "code", "source": ["print('nothing to patch')\n"]}]}))


if __name__ == "__main__":
    unittest.main()
