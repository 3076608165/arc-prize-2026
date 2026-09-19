"""Which cheap re-ranking recovers the outputs the shipped selector mis-ranks?

Measured, 2026-09-19, on the saved full-eval store at `agi2/out/v7/inference_outputs`
(144 outputs, 141 with ground truth, median 17 candidates each):

    score_full_probmul_3     42/141   29.8%   <- what the kernel ships
    score_kgmon              41/141   29.1%
    score_cellvote           36/141   25.5%
    ORACLE (any candidate)   51/141   36.2%

Nine outputs are already decoded correctly and merely ranked below two wrong
guesses. No new decode, no submission slot: this script ranks the same pool with
constraints derived from the task's own training pairs, plus an ensemble of the
existing selectors, and prints what each recovers.

    python agi2/selector_search.py                 # every selector, v7 store
    python agi2/selector_search.py --list
    python agi2/selector_search.py --self-test

The constraints are filters, not scores: a candidate that contradicts a rule the
training pairs all obey is dropped, then the pool is ranked as before. Every
filter falls back to the unfiltered pool when it would empty it -- a rule
inferred from three examples is a heuristic, and a heuristic that cannot lose is
worth less than one that can.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_offline as SO  # noqa: E402
from arclib.loader import load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --- constraints inferred from a task's training pairs -----------------------

def shape_rule(task):
    """What the output shape must be, in terms of the test input's shape."""
    same = [np.shape(p.output) == np.shape(p.input) for p in task.train]
    if all(same):
        return lambda test_input: tuple(np.shape(test_input))
    shapes = {np.shape(p.output) for p in task.train}
    if len(shapes) == 1:
        fixed = shapes.pop()
        return lambda test_input: fixed
    factors = set()
    for p in task.train:
        hi, ho = np.shape(p.input), np.shape(p.output)
        if hi[0] and hi[1] and ho[0] % hi[0] == 0 and ho[1] % hi[1] == 0:
            factors.add((ho[0] // hi[0], ho[1] // hi[1]))
        else:
            return None
    if len(factors) == 1:
        f = factors.pop()
        return lambda test_input: (np.shape(test_input)[0] * f[0], np.shape(test_input)[1] * f[1])
    return None


def palette_rule(task):
    """True when every training output uses only colours its input already had."""
    for p in task.train:
        if not set(np.unique(p.output)) <= set(np.unique(p.input)):
            return False
    return True


def common_symmetries(task):
    """D4 elements every training output has (the rule the field calls 'symmetry')."""
    syms = []
    for name, fn in (("rot180", np.rot90), ("rot180x2", lambda g: np.rot90(g, 2)),
                     ("fliplr", np.fliplr), ("flipud", np.flipud),
                     ("transpose", np.transpose), ("antitranspose", lambda g: np.rot90(g, 2).T)):
        if all(np.array_equal(fn(np.asarray(p.output)), np.asarray(p.output)) for p in task.train):
            syms.append((name, fn))
    return syms


def build_filters(task, test_input):
    """A list of named predicates a candidate output must satisfy."""
    out = []
    rule = shape_rule(task)
    if rule is not None:
        shape = rule(test_input)
        out.append(("shape", lambda g, s=shape: tuple(np.shape(g)) == s))
    if palette_rule(task):
        allowed = set(np.unique(test_input))
        out.append(("palette", lambda g, a=allowed: set(np.unique(g)) <= a))
    syms = common_symmetries(task)
    if syms:
        out.append(("symmetry", lambda g, s=[fn for _, fn in syms]:
                    all(np.array_equal(fn(np.asarray(g)), np.asarray(g)) for fn in s)))
    return out


# --- selectors ---------------------------------------------------------------

def borda(algos):
    """Rank by summed rank across selectors: the cheap ensemble."""
    def selector(guesses):
        total: dict[tuple, float] = {}
        grid: dict[tuple, object] = {}
        for algo in algos:
            for rank, g in enumerate(algo(guesses)):
                h = tuple(map(tuple, np.asarray(g)))
                total[h] = total.get(h, 0.0) + rank
                grid[h] = g
        return [grid[h] for h in sorted(total, key=lambda h: total[h])]
    selector.__name__ = "borda(" + "+".join(a.__name__ for a in algos) + ")"
    return selector


# --- self-test ---------------------------------------------------------------

def self_test() -> None:
    """The constraint inference is the only non-obvious logic here."""
    from arclib.loader import Pair, Task
    task = Task("t", [
        Pair(input=[[1, 2], [3, 4]], output=[[1, 2, 1, 2], [3, 4, 3, 4],
                                             [1, 2, 1, 2], [3, 4, 3, 4]]),
    ], [Pair(input=[[5, 6], [7, 8]], output=None)])
    test_input = [[5, 6], [7, 8]]
    filters = dict(build_filters(task, test_input))
    assert set(filters) == {"shape", "palette"}, filters
    assert filters["shape"]([[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]])
    assert not filters["shape"]([[5, 6], [7, 8]])            # 2x2 is not 2x-upscaled
    assert filters["palette"]([[5, 6, 5, 6]] * 4)
    assert not filters["palette"]([[5, 6, 9, 6]] * 4)        # 9 never appears in the input

    sym = Task("s", [Pair(input=[[1]], output=[[1, 2, 1], [3, 4, 3], [1, 2, 1]])], [])
    names = [n for n, _ in common_symmetries(sym)]
    assert "fliplr" in names and "rot180" not in names, names
    assert dict(build_filters(sym, [[0]]))["symmetry"]([[1, 2, 1], [3, 4, 3], [1, 2, 1]])
    assert not dict(build_filters(sym, [[0]]))["symmetry"]([[1, 2, 1], [3, 4, 3], [1, 5, 1]])
    print("self-test ok: shape/palette/symmetry rules admit the right candidates")


# --- main --------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=ROOT / "agi2/out/v7/inference_outputs")
    ap.add_argument("--decoder", type=Path, default=ROOT / "agi2/out/v7/arc_decoder.py")
    ap.add_argument("--mix-store", type=Path, default=None,
                    help="score a second store on the shared outputs: does one "
                         "attempt from each config beat two attempts from one?")
    ap.add_argument("--mix-decoder", type=Path, default=None)
    ap.add_argument("--n-guesses", type=int, default=2)
    ap.add_argument("--list", action="store_true", help="list selectors and exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    dec = SO.load_decoder(args.decoder)
    store = SO.load_store(args.store)
    truth = SO.ground_truth()
    tasks = {t.task_id: t for t in load_split("evaluation")}

    base_algos = list(getattr(dec, "selection_algorithms", []))

    # One filter set per output, built from that task's training pairs.
    filters = {}
    for base in store:
        task_id, _, idx = base.rpartition("_")
        try:
            task = tasks[task_id]
            test_input = np.asarray(task.test[int(idx)].input)
            filters[base] = build_filters(task, test_input)
        except (KeyError, IndexError, ValueError):
            # A store can hold outputs the public split does not describe
            # (different task ids or test indices); those get ranked unfiltered.
            filters[base] = []

    # The selector API sees only the pool, so filters ride along per output via
    # a closure over a mutable current-filters variable.
    state = {"filters": []}

    def with_filters(algo):
        def selector(guesses):
            ranked = algo(guesses)
            if not state["filters"]:
                return ranked
            kept = [g for g in ranked if all(f(np.asarray(g)) for _, f in state["filters"])]
            if kept:
                state["applied"] = state.get("applied", 0) + 1
                if len(kept) < len(ranked):
                    state["shrank"] = state.get("shrank", 0) + 1
                return kept
            return ranked
        selector.__name__ = algo.__name__ + " +filters"
        return selector

    variants = []
    for algo in base_algos:
        variants.append((algo.__name__, algo, False, algo))
        variants.append((algo.__name__ + " +filters", with_filters(algo), True, algo))
    if len(base_algos) >= 2:
        ens = borda(base_algos)
        variants.append((ens.__name__, ens, False, ens))
        variants.append((ens.__name__ + " +filters", with_filters(ens), True, ens))

    if args.list:
        for name, *_ in variants:
            print(name)
        return 0

    print(f"store   : {args.store}")
    scored = [b for b in store if b in truth]
    print(f"outputs : {len(scored)} with ground truth ({len(store)} in store), "
          f"top-{args.n_guesses} counts as solved\n")

    order = [b for b in sorted(store)]
    rows = []
    for name, algo, use_filters, _ in variants:
        state.clear()
        solved = total = 0
        shrank = kept_pool = 0
        for base in order:
            if base not in truth:
                continue
            total += 1
            state["filters"] = filters.get(base, []) if use_filters else []
            try:
                ranked = algo(store[base])
            except Exception as exc:
                print(f"  {name}: {base} raised {type(exc).__name__}: {exc}")
                continue
            if use_filters and state.get("shrank"):
                shrank += 1
            if state.get("applied"):
                kept_pool += 1
            if any(np.array_equal(np.asarray(g), truth[base]) for g in ranked[:args.n_guesses]):
                solved += 1
        rows.append((name, solved, total, shrank))

    rows.append(("ORACLE (any candidate)", *SO.evaluate(store, truth, lambda g: [s["solution"] for s in g.values()], n_guesses=10**6)[:2], 0))
    base_solved = rows[0][1]
    width = max(len(r[0]) for r in rows)
    for name, solved, total, shrank in rows:
        delta = "" if name == rows[0][0] else f"  ({solved - base_solved:+d})"
        extra = f"   [pool shrank on {shrank} outputs]" if shrank else ""
        print(f"  {name:<{width}}  {solved:>3}/{total}  {solved / total:6.1%}{delta}{extra}")

    if args.mix_store:
        mix_main(store, args.mix_store, args.mix_decoder or args.decoder, truth)
    return 0


def mix_main(store_a: dict, store_b_path: Path, decoder_b: Path, truth: dict) -> None:
    """Two attempts from one config vs one attempt from each of two configs.

    Same outputs, same Kaggle rule, same selector: the only difference is where
    the second guess comes from. A split budget costs per-task samples, so this
    is the upside half of that trade -- it says nothing about the cost half.
    """
    store_b = SO.load_store(store_b_path)
    dec_a = SO.load_decoder(ROOT / "agi2/out/v7/arc_decoder.py")
    dec_b = SO.load_decoder(decoder_b)
    algo_a = next(a for a in dec_a.selection_algorithms if a.__name__ == "score_full_probmul_3")
    algo_b = next((a for a in dec_b.selection_algorithms if a.__name__ == "score_full_probmul_3"), algo_a)
    shared = sorted(set(store_a) & set(store_b) & set(truth))
    if not shared:
        print(f"\nmix: no shared outputs with {store_b_path}")
        return

    both = same_both = 0
    a_only = b_only = 0
    for base in shared:
        want = truth[base]
        top_a = np.asarray(algo_a(store_a[base])[0])
        top_b = np.asarray(algo_b(store_b[base])[0])
        two_a = [np.asarray(g) for g in algo_a(store_a[base])[:2]]
        ok_a = any(np.array_equal(g, want) for g in two_a)
        ok_mix = np.array_equal(top_a, want) or np.array_equal(top_b, want)
        both += ok_a
        same_both += ok_mix
        if ok_mix and not ok_a:
            a_only += 1
        if ok_a and not ok_mix:
            b_only += 1
    n = len(shared)
    print(f"\nmix on {n} shared outputs ({store_b_path.name}):")
    print(f"  top-2 from config A alone        {both:>3}/{n}  {both / n:6.1%}")
    print(f"  top-1 from A + top-1 from B      {same_both:>3}/{n}  {same_both / n:6.1%}"
          f"  ({same_both - both:+d})")
    print(f"  gained by B / lost by dropping A's 2nd guess:  +{a_only} / -{b_only}")


if __name__ == "__main__":
    raise SystemExit(main())
