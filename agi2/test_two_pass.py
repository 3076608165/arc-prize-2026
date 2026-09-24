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

import numpy as np

from repro_nvarc import (PASS2_STORE, add_two_pass_pooling,
                         drop_allowlist, fetch_upstream, order_largest_first)

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
        """Both passes carry the same filter, and the pass id rides on the item.

        A `__PASS2__` marker in the queue can only be consumed by one worker, so
        three of four workers used to re-decode pass B's copies with pass A's
        seeds. The pass id has to travel with the key.
        """
        nb = patched()
        starter = cell_with(nb, "_queue_keys = sorted")
        loops = starter.count("for key in _queue_keys:")
        self.assertEqual(loops, 2)
        self.assertEqual(starter.count('queue.put("0934a4d8")'), 0)  # never hardcoded
        allowlist = 'if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:'
        self.assertEqual(starter.count(allowlist), 2)
        self.assertEqual(starter.count("        queue.put((key, 0))\n"), 1)
        self.assertEqual(starter.count("        queue.put((key, 1))\n"), 1)
        self.assertNotIn("__PASS2__", starter)
        self.assertLess(starter.index("_queue_keys = sorted"),
                        starter.index("for key in _queue_keys:"))
        self.assertLess(starter.index("queue.put((key, 0))"),
                        starter.index("queue.put((key, 1))"))

    def test_after_full_eval_the_second_loop_has_no_filter(self):
        nb = upstream_notebook()
        self.assertTrue(drop_allowlist(nb))
        self.assertTrue(add_two_pass_pooling(nb))
        starter = cell_with(nb, "_queue_keys = sorted")
        self.assertEqual(starter.count("for key in _queue_keys:"), 2)
        self.assertNotIn("continue", starter.split("queue.put((key, 0))")[1])
        self.assertEqual(starter.count("        queue.put((key, 1))\n"), 1)

    def test_worker_routes_each_item_to_its_own_pass(self):
        solver = written_files(patched())["arc_solver.py"]
        self.assertIn("key, pass_id = item", solver)
        self.assertIn("dir_outputs = _pass_store(pass_id)", solver)
        self.assertEqual(solver.count("os.makedirs(dir_outputs, exist_ok=True)"), 2)
        self.assertNotIn("__PASS2__", solver)
        self.assertIn("seed=1 + 101 * pass_id", solver)
        self.assertIn("seed=2 + 103 * pass_id", solver)

    def test_pass_store_sends_pass_b_to_its_own_store(self):
        """Execute the emitted mapping: pass A's store is the one already read."""
        solver = written_files(patched())["arc_solver.py"]
        fn = next(node for node in ast.walk(ast.parse(solver))
                  if isinstance(node, ast.FunctionDef) and node.name == "_pass_store")
        ns: dict = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<solver>", "exec"), ns)
        self.assertEqual(ns["_pass_store"](0), "/kaggle/inference_outputs")
        self.assertEqual(ns["_pass_store"](1), PASS2_STORE)

    def test_selection_is_monotone(self):
        nb = patched()
        selection = cell_with(nb, "data.get_submission(selected)")
        self.assertIn(f'os.path.isdir("{PASS2_STORE}")', selection)
        self.assertIn("attempt_1 pinned to pass A's top-1", selection)
        self.assertIn("def _pin_pass_a(pooled, a_only)", selection)
        self.assertIn("hashable(c) not in seen", selection)
        self.assertIn("selected = _pin_pass_a(selected, _a_only)", selection)

    def test_pinned_merge_runs_and_keeps_pass_a_first(self):
        """Execute the emitted merge on grids -- the shape it really receives.

        `score_kgmon` returns lists of *grids*, not sample dicts. The first
        version of this cell wrote `if _first`, which asks a grid for its truth
        value: a grid with more than one cell makes that a ValueError, and the
        push died there 43 minutes into its first cell. Source-text assertions
        cannot see that, so this one runs the emitted function.
        """
        selection = cell_with(patched(), "def _pin_pass_a(pooled, a_only)")
        fn = next(node for node in ast.walk(ast.parse(selection))
                  if isinstance(node, ast.FunctionDef) and node.name == "_pin_pass_a")
        ns: dict = {"hashable": lambda g: tuple(map(tuple, g))}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<cell>", "exec"), ns)
        pin = ns["_pin_pass_a"]
        a, b, c = np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), 7)
        # pass B's ranking promotes b over a; attempt_1 stays pass A's top-1.
        merged = pin({"k": [b, a]}, {"k": [a]})
        self.assertEqual(len(merged["k"]), 2)
        self.assertTrue(np.array_equal(merged["k"][0], a))
        self.assertTrue(np.array_equal(merged["k"][1], b))
        # A task pass A never reached keeps pass B's own ranking, unpinned.
        untouched = pin({"k": [c]}, {})
        self.assertEqual([g.tolist() for g in untouched["k"]], [c.tolist()])
        # And the single-cell grid that a `len`-based check would wave through.
        tiny = pin({"k": [np.zeros((1, 1))]}, {"k": [np.zeros((1, 1))]})
        self.assertEqual(len(tiny["k"]), 1)

    def test_patched_files_compile(self):
        for name, src in written_files(patched()).items():
            if name.endswith(".py"):
                ast.parse(src, filename=name)

    def test_is_all_or_nothing_on_a_notebook_without_the_anchors(self):
        self.assertFalse(add_two_pass_pooling({"cells": [
            {"cell_type": "code", "source": ["print('nothing to patch')\n"]}]}))


    def test_largest_first_reorders_the_queue_and_nothing_else(self):
        """Scheduling only: same key list, expensive first, pool untouched.

        The cost proxy runs in the starter process against the challenge shape
        ({"test": [{"input": [[...]]}]}), so it is evaluated here on that shape
        rather than assumed.
        """
        nb = upstream_notebook()
        self.assertTrue(add_two_pass_pooling(nb))
        self.assertTrue(order_largest_first(nb))
        starter = cell_with(nb, "Largest puzzles first")
        self.assertNotIn("sorted(data.keys())", starter)
        self.assertEqual(starter.count("for key in _queue_keys:"), 2)
        expr = [ln.strip() for ln in starter.splitlines()
                if ln.strip().startswith("key=lambda k:")][0][len("key="):-1]
        fake = {"small": {"test": [{"input": [[0] * 3] * 3}]},
                "big": {"test": [{"input": [[0] * 30] * 30}]}}
        # eval is deliberate and test-local: the expression is a literal slice of
        # our own patch string (no external input), and evaluating it verbatim is
        # the point -- it is the expression the notebook will run.
        key = eval(expr, {"data": fake})
        self.assertEqual(sorted(fake, key=key), ["big", "small"])


if __name__ == "__main__":
    unittest.main()
