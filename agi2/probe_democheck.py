"""Can the demo pairs veto a wrong candidate? Measure it -- offline, no quota.

The pool oracle on the saved v7 store is 51/141 while upstream's ranking submits
42/141, and three re-rankings of the same per-view scores all failed to touch
that gap (mean/median rank, z-score, view votes: 41, 42, 34). So the missing nine
are not a scoring artifact -- they are candidates the model's own plausibility
score cannot distinguish from the wrong ones.

A *verifier* is a different thing from a ranker: it does not ask "which candidate
looks likelier", it asks "can this candidate be ruled out". The demo pairs answer
that without any model, because a candidate that breaks an invariant the demos
observe is wrong regardless of how confident the model was. This measures three
such invariants, applied only as a veto on top of upstream's order:

  colour closure   a colour no demo pair ever shows cannot appear in the answer
  shape            the answer's shape must be one the demos' rule can produce
  symmetry         a transform that fixes every demo input AND output, and fixes
                   the test input, should fix the answer too

    python agi2/probe_democheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_offline as SO  # noqa: E402
from arclib.loader import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "agi2" / "out" / "v7"


def transforms():
    """D4: eight maps, identity first."""
    out = []
    for k in range(4):
        out.append(lambda g, k=k: np.rot90(g, k))
    out.append(lambda g: np.transpose(g))
    for k in range(1, 4):
        out.append(lambda g, k=k: np.rot90(np.transpose(g), k))
    return out


T = transforms()


def stable_under(g, t) -> bool:
    try:
        return np.array_equal(t(g), g)
    except Exception:
        return False


def demo_invariants(train):
    """(colours seen, output shapes seen, transforms that fix every demo grid)."""
    grids_in = [np.asarray(p.input) for p in train]
    grids_out = [np.asarray(p.output) for p in train]
    colours = {int(v) for g in grids_in + grids_out for v in np.unique(g)}
    shapes = {g.shape for g in grids_out}
    syms = []
    for i, t in enumerate(T):
        if i == 0:
            continue
        if all(stable_under(g, t) for g in grids_in + grids_out):
            syms.append(t)
    return colours, shapes, syms


def violations(grid, colours, shapes, syms, test_in) -> int:
    g = np.asarray(grid)
    n = 0
    if not {int(v) for v in np.unique(g)} <= colours:
        n += 1
    if g.shape not in shapes and g.shape != test_in.shape:
        n += 1
    for t in syms:
        if stable_under(test_in, t) and not stable_under(g, t):
            n += 1
    return n


def main() -> int:
    dec = SO.load_decoder(STORE / "arc_decoder.py", with_cellvote=False)
    store = SO.load_store(STORE / "inference_outputs")
    truth = SO.ground_truth()
    tasks = {}
    for t in load_split("evaluation"):
        for i, pair in enumerate(t.test):
            tasks[f"{t.task_id}_{i}"] = (t.train, np.asarray(pair.input))
    bases = sorted(b for b in store if b in truth and b in tasks)

    base = veto = 0
    gained, lost = [], []
    veto_fired = 0
    for b in bases:
        train, test_in = tasks[b]
        colours, shapes, syms = demo_invariants(train)
        want = truth[b]
        upstream = [np.asarray(x) for x in dec.score_full_probmul_3(store[b])]
        hit = lambda gs: any(np.array_equal(g, want) for g in gs)
        base += hit(upstream[:2])

        ranked = sorted(range(len(upstream)),
                        key=lambda i: (violations(upstream[i], colours, shapes, syms, test_in), i))
        reordered = [upstream[i] for i in ranked]
        if ranked != sorted(range(len(upstream))):
            veto_fired += 1
        got = hit(reordered[:2])
        veto += got
        if got and not hit(upstream[:2]):
            gained.append(b)
        if hit(upstream[:2]) and not got:
            lost.append(b)

    n = len(bases)
    print(f"outputs scored: {n}\n")
    print(f"  upstream top-2 (today)          {base:>3}/{n}  {base/n:6.1%}")
    print(f"  + demo veto                     {veto:>3}/{n}  {veto/n:6.1%}  ({veto-base:+d})")
    print(f"  the veto reordered {veto_fired}/{n} outputs")
    print(f"  gained {gained}")
    print(f"  lost   {lost}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
