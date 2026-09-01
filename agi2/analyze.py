"""Why does the search score 4.8% on training and exactly 0% on evaluation?

"Zero" is a suspicious number, so this separates the possible explanations.
For each task we ask how far the best searched program gets:

    solved        every demo output reproduced exactly
    shape-only    every demo output's *shape* reproduced, contents wrong
    partial       right shape and >=90% of cells right
    nothing       no program in the space even matches the output shapes

If evaluation were merely harder we would expect the shape-only and partial
counts to stay healthy while `solved` fell. If the whole hypothesis class is
simply wrong for ARC-AGI-2, all four collapse together.

    python agi2/analyze.py --split evaluation
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arclib.dsl import PRIMITIVES
from arclib.grids import shape
from arclib.loader import load_split
from arclib.search import _apply, _key

MAX_NODES = 600


def best_effort(demos, max_depth: int = 3) -> tuple[str, float]:
    """Best (category, cell accuracy) any program up to max_depth achieves."""
    ins = [p.input for p in demos]
    outs = [p.output for p in demos]

    def grade(got: list) -> tuple[bool, bool, float]:
        exact = got == outs
        shapes_ok = all(shape(a) == shape(b) for a, b in zip(got, outs))
        if not shapes_ok:
            return exact, False, 0.0
        tot = hit = 0
        for a, b in zip(got, outs):
            for ra, rb in zip(a, b):
                for x, y in zip(ra, rb):
                    tot += 1
                    hit += x == y
        return exact, True, hit / tot if tot else 0.0

    best_shape = False
    best_acc = 0.0
    exact, sh, acc = grade(ins)
    best_shape, best_acc = sh, acc
    if exact:
        return "solved", 1.0

    frontier = [((), ins)]
    seen = {_key(ins)}
    for _ in range(max_depth):
        nxt = []
        for prog, grids in frontier:
            for prim in PRIMITIVES:
                res = _apply(prim, grids)
                if res is None:
                    continue
                k = _key(res)
                if k in seen:
                    continue
                seen.add(k)
                exact, sh, acc = grade(res)
                if exact:
                    return "solved", 1.0
                best_shape = best_shape or sh
                best_acc = max(best_acc, acc)
                nxt.append((prog + (prim[0],), res))
                if len(nxt) >= MAX_NODES:
                    break
            if len(nxt) >= MAX_NODES:
                break
        frontier = nxt
        if not frontier:
            break

    if best_acc >= 0.9:
        return "partial", best_acc
    if best_shape:
        return "shape-only", best_acc
    return "nothing", best_acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--depth", type=int, default=3)
    args = ap.parse_args()

    tasks = load_split(args.split)
    if args.limit:
        tasks = tasks[: args.limit]

    t0 = time.perf_counter()
    cats = Counter()
    accs = []
    for t in tasks:
        cat, acc = best_effort(t.train, args.depth)
        cats[cat] += 1
        accs.append(acc)
    dt = time.perf_counter() - t0

    n = len(tasks)
    print(f"{args.split}: {n} tasks, depth {args.depth}, {dt:.0f}s\n")
    for cat in ("solved", "partial", "shape-only", "nothing"):
        c = cats[cat]
        print(f"  {cat:<12} {c:>4}  {c / n:6.1%}")
    accs.sort()
    print(f"\n  best cell accuracy: median {accs[n // 2]:.1%}, "
          f"p90 {accs[int(n * 0.9)]:.1%}, max {accs[-1]:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
