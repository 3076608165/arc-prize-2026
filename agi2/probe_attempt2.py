"""Is the second attempt being spent on a redraw? Measure it before building anything.

`score_full_probmul_3` ranks candidates and emits the top two. Nothing in that
ranking says the two must be *different hypotheses*, and agi2/cellvote_patch.py
already measured that attempt_2 alone rescues 7 of 41 solved outputs -- so the
second slot carries real value whenever it holds a different answer, and none
when it holds the same answer again.

This asks the cheap question first, on the saved store, at zero quota:

    baseline        upstream's top-2 (what we submit today)
    spread(t)       attempt_2 = the first candidate that disagrees with
                    attempt_1 in more than t of its cells
    oracle(topk)    the best a spread from the first k ranked candidates
                    could do, i.e. this idea's ceiling

    python agi2/probe_attempt2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_offline as SO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "agi2" / "out" / "v7"


def disagreement(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(a != b))


def main() -> int:
    dec = SO.load_decoder(STORE / "arc_decoder.py", with_cellvote=False)
    store = SO.load_store(STORE / "inference_outputs")
    truth = SO.ground_truth()
    bases = sorted(b for b in store if b in truth)

    tol_lo, tol_hi = 0.0, 0.0
    rows = {t: 0 for t in (0.001, 0.02, 0.05, 0.10, 0.25)}
    ceilings = {k: 0 for k in (2, 3, 5, 10)}
    base_solved = 0
    top2_same = 0

    for b in bases:
        ranked = [np.asarray(x) for x in dec.score_full_probmul_3(store[b])]
        want = truth[b]
        if not ranked:
            continue
        a1 = ranked[0]

        def hit(grids, want=want):
            return any(np.array_equal(np.asarray(g), want) for g in grids)

        base_solved += hit(ranked[:2])
        if len(ranked) > 1 and disagreement(ranked[0], ranked[1]) <= 0.02:
            top2_same += 1
        for tol in rows:
            a2 = next((x for x in ranked[1:] if disagreement(a1, x) > tol), a1)
            rows[tol] += hit((a1, a2))
        for k in ceilings:
            head = ranked[:k]
            a2 = next((x for x in head[1:] if disagreement(a1, x) > 0.02), a1)
            ceilings[k] += hit((a1, a2))

    n = len(bases)
    print(f"outputs with ground truth: {n}\n")
    print(f"  upstream top-2 (today)          {base_solved:>3}/{n}  {base_solved/n:6.1%}")
    print(f"  top-2 are within 2% of cells    {top2_same:>3}/{n}  {top2_same/n:6.1%}"
          f"   <- slots spent on a redraw")
    for tol, hits in rows.items():
        print(f"  spread, attempt_2 disagrees >{tol:<5} {hits:>3}/{n}  {hits/n:6.1%}"
              f"  ({hits - base_solved:+d})")
    print("\n  ceiling of a spread from the first k candidates:")
    for k, hits in ceilings.items():
        print(f"    k={k:<3}                          {hits:>3}/{n}  ({hits - base_solved:+d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
