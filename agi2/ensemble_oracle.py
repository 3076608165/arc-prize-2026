"""Is a second model worth building an ensemble around? Decide offline first.

The 2025 winner's structural difference from the public recipe is an ensemble.
Before spending days wiring one up under a 12h wall, ask the cheap question:
do the 2B model's candidates contain correct answers the 4B's do not?

Three numbers per shared output, all from saved stores and ground truth:

    oracle(4B)      any 4B candidate correct
    oracle(2B)      any 2B candidate correct
    oracle(4B|2B)   any candidate from either  <- the ensemble ceiling

and one selector:

    cross-agree     grids produced by BOTH models, ranked by upstream's score
                    over the merged pool -- the actual ensemble mechanism

If oracle(4B|2B) is not clearly above oracle(4B), the ensemble has no
headroom and the idea dies here at zero build cost.

    python agi2/ensemble_oracle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_offline as SO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
A = ("4B", ROOT / "agi2" / "out" / "v8")
B = ("2B", ROOT / "agi2" / "out" / "exp-ep2" / "result")  # pulled store; exp-2b/result holds no inference_outputs


def load(d: Path):
    if not (d / "inference_outputs").is_dir():
        return None, None
    return SO.load_decoder(d / "arc_decoder.py", with_cellvote=False), SO.load_store(d / "inference_outputs")


def correct_in(store, base, truth):
    return any(np.array_equal(np.asarray(g["solution"]), truth[base]) for g in store[base].values())


def main() -> int:
    truth = SO.ground_truth()
    da, sa = load(A[1])
    db, sb = load(B[1])
    if sa is None or sb is None:
        raise SystemExit(f"need both stores: {A[1]} and {B[1]}")
    shared = sorted(b for b in sa if b in sb and b in truth)
    n = len(shared)
    print(f"shared scoreable outputs: {n}\n")

    oa = {b for b in shared if correct_in(sa, b, truth)}
    ob = {b for b in shared if correct_in(sb, b, truth)}
    print(f"  oracle({A[0]})        {len(oa):>3}/{n}")
    print(f"  oracle({B[0]})        {len(ob):>3}/{n}")
    print(f"  oracle({A[0]}|{B[0]})     {len(oa | ob):>3}/{n}   <- ensemble ceiling")
    print(f"  only {B[0]} has it     {len(ob - oa):>3}      <- what the ensemble could add")
    print(f"  only {A[0]} has it     {len(oa - ob):>3}")

    # upstream selector on each alone, and on the merged pool
    def top2(dec, store, b):
        return [np.asarray(x) for x in dec.score_full_probmul_3(store[b])[:2]]

    sel_a = {b for b in shared if any(np.array_equal(x, truth[b]) for x in top2(da, sa, b))}
    sel_b = {b for b in shared if any(np.array_equal(x, truth[b]) for x in top2(db, sb, b))}
    merged = {b: {**{f"a.{k}": v for k, v in sa[b].items()}, **{f"b.{k}": v for k, v in sb[b].items()}} for b in shared}
    sel_m = {b for b in shared if any(np.array_equal(x, truth[b]) for x in top2(da, merged, b))}

    # cross-agreement: prefer grids both models produced
    def cross(b):
        ha = {da.hashable(g["solution"]) for g in sa[b].values()}
        hb = {db.hashable(g["solution"]) for g in sb[b].values()}
        both = ha & hb
        ranked = da.score_full_probmul_3(merged[b])
        agreed = [x for x in ranked if da.hashable(x) in both]
        rest = [x for x in ranked if da.hashable(x) not in both]
        return [np.asarray(x) for x in (agreed + rest)[:2]]
    sel_x = {b for b in shared if any(np.array_equal(x, truth[b]) for x in cross(b))}

    print(f"\n  selector, {A[0]} alone       {len(sel_a):>3}/{n}")
    print(f"  selector, {B[0]} alone       {len(sel_b):>3}/{n}")
    print(f"  selector, merged pool     {len(sel_m):>3}/{n}")
    print(f"  cross-agree first         {len(sel_x):>3}/{n}   <- the ensemble mechanism")

    # Every selector above answers "which ONE grid do I submit". The competition
    # does not ask that: arclib/evaluate.py counts an output correct if EITHER
    # attempt matches, so attempt_2 is free real estate and no selector is
    # involved. The measurable question is therefore not "can a selector reach
    # the oracle" (it cannot) but "is a DIFFERENT MODEL's top-1 a better use of
    # the second slot than this model's own top-2" -- i.e. are the two models'
    # errors less correlated than one model's first and second guess.
    swap = {b for b in shared
            if any(np.array_equal(x, truth[b]) for x in (
                [np.asarray(y) for y in da.score_full_probmul_3(sa[b])[:1]]
                + [np.asarray(y) for y in db.score_full_probmul_3(sb[b])[:1]]))}
    print(f"\n  attempts = {A[0]} top1+top2 (today) {len(sel_a):>3}/{n}")
    print(f"  attempts = {A[0]} top1, {B[0]} top1     {len(swap):>3}/{n}   <- the second slot, spent on the other model")
    print(f"  gained {sorted(swap - sel_a)}")
    print(f"  lost   {sorted(sel_a - swap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
