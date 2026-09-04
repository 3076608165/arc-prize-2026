"""Stratify a submission's failures against the public evaluation ground truth.

`analyze.py` asks how close our *search space* gets. This asks how close an
actual submission gets, which is the question that matters once a real model is
producing the answers.

The distinction that drives everything: ARC pays only for exact matches, so a
prediction with the right shape and two wrong cells scores the same as a blank
grid. If a model's failures are mostly near-misses, a verify-and-repair stage
sits on top of it and converts them. If its failures are structurally wrong,
repair has nothing to work with and the effort belongs elsewhere.

    python agi2/analyze_submission.py agi2/out/nvfull/submission.json
    python agi2/analyze_submission.py <path> --show 8      # render worst near-misses
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arclib.grids import shape
from arclib.loader import load_split
from arclib.viz import task_to_html, write_html


def classify(pred, truth) -> tuple[str, float]:
    """Return (category, cell accuracy) for one prediction against one truth."""
    if pred is None or not pred or not pred[0]:
        return "empty", 0.0
    if pred == truth:
        return "solved", 1.0
    if shape(pred) != shape(truth):
        return "wrong-shape", 0.0
    tot = hit = 0
    for rp, rt in zip(pred, truth):
        for a, b in zip(rp, rt):
            tot += 1
            hit += a == b
    acc = hit / tot if tot else 0.0
    if acc >= 0.99:
        return "off-by-a-few", acc
    if acc >= 0.90:
        return "near-miss", acc
    return "right-shape-only", acc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", type=Path)
    ap.add_argument("--split", default="evaluation")
    ap.add_argument("--show", type=int, default=0,
                    help="render this many of the closest failures to HTML")
    ap.add_argument("--out", default="agi2/out/near_misses.html")
    args = ap.parse_args()

    sub = json.loads(args.submission.read_text(encoding="utf-8"))
    tasks = {t.task_id: t for t in load_split(args.split)}

    cats: Counter = Counter()
    accs: list[float] = []
    closest: list[tuple[float, str, int, list, list]] = []
    n = 0

    for tid, entry in sub.items():
        task = tasks.get(tid)
        if task is None:
            continue
        for i, att in enumerate(entry):
            if i >= len(task.test):
                break
            truth = task.test[i].output
            if truth is None:
                continue
            n += 1
            # Score the output the way Kaggle does -- best of the two attempts.
            # `best` starts unset rather than at ("empty", 0.0): a genuine
            # wrong-shape guess also scores 0.0, and a `>` comparison against a
            # 0.0 sentinel would never replace it, reporting every failure as
            # "empty".
            best: tuple[str, float, list | None] | None = None
            for key in ("attempt_1", "attempt_2"):
                cat, acc = classify(att.get(key), truth)
                if cat == "solved":
                    best = (cat, acc, att.get(key))
                    break
                if best is None or acc > best[1]:
                    best = (cat, acc, att.get(key))
            assert best is not None
            cats[best[0]] += 1
            accs.append(best[1])
            if best[0] not in ("solved", "empty"):
                closest.append((best[1], tid, i, best[2], truth))

    if not n:
        print("no scored outputs -- is this the right split?")
        return 1

    print(f"{args.submission}  vs  {args.split}  ({n} outputs)\n")
    order = ["solved", "off-by-a-few", "near-miss", "right-shape-only",
             "wrong-shape", "empty"]
    for cat in order:
        c = cats[cat]
        bar = "#" * round(40 * c / n)
        print(f"  {cat:<18} {c:>4}  {c / n:6.1%}  {bar}")

    accs.sort()
    print(f"\n  cell accuracy: median {accs[len(accs) // 2]:.1%}, "
          f"p90 {accs[int(len(accs) * 0.9)]:.1%}")

    repairable = cats["off-by-a-few"] + cats["near-miss"]
    print(f"\n  solved                     : {cats['solved']}/{n} "
          f"({cats['solved'] / n:.1%})")
    print(f"  within reach of a repair   : {repairable}/{n} "
          f"({repairable / n:.1%})")
    if cats["solved"]:
        print(f"  ceiling if every near-miss were repaired: "
              f"{(cats['solved'] + repairable) / n:.1%} "
              f"({(cats['solved'] + repairable) / max(cats['solved'], 1):.2f}x current)")

    if args.show and closest:
        closest.sort(key=lambda t: -t[0])
        picked = closest[: args.show]
        body = []
        for acc, tid, i, pred, truth in picked:
            t = tasks[tid]
            body.append(task_to_html(
                t, predictions=[[pred] if j == i else [] for j in range(len(t.test))],
                title=f"{tid} test{i} -- {acc:.1%} of cells correct"))
        p = write_html("<hr style='border-color:#333;margin:32px 0'>".join(body),
                       args.out, title="closest failures")
        print(f"\n  wrote {p.resolve()}  ({len(picked)} closest failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
