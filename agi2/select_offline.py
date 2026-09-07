"""Compare selection algorithms offline, against the saved candidate store.

The expensive half of the NVARC pipeline is decoding: per-task LoRA test-time
training plus a constrained DFS, ~8 worker-minutes a task. The cheap half is
deciding which of the resulting candidates to submit. Those are separable, and
only the second half is where our own ideas live -- so `repro_nvarc.py
--save-candidates` copies the decode store out of the kernel once, and every
selection experiment after that runs here in seconds at zero quota.

    python agi2/select_offline.py                       # compare all selectors
    python agi2/select_offline.py --store <dir>
    python agi2/select_offline.py --only score_cellvote --show 6

The store is what `ArcDecoder.load_decoded_results` reads: one bz2-pickled file
per (task, augmentation), named `<task>_<test_index>.<augmentations>`, each
holding a list of samples with `solution`, `beam_score` and `score_aug`.
"""
from __future__ import annotations

import argparse
import bz2
import importlib.util
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arclib.loader import load_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = ROOT / "agi2" / "out" / "v7" / "inference_outputs"
DEFAULT_DECODER = ROOT / "agi2" / "out" / "v7" / "arc_decoder.py"


def load_decoder(path: Path, with_cellvote: bool = True):
    """Import the decoder the kernel ran, then splice in our own selectors.

    Injecting here rather than requiring the kernel to have been built with
    --cellvote means a single saved store serves every selector we invent
    later, including ones written after the decode was paid for.
    """
    spec = importlib.util.spec_from_file_location("arc_decoder_run", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if with_cellvote and not hasattr(mod, "score_cellvote"):
        from cellvote_patch import CELLVOTE_SRC

        exec(compile(CELLVOTE_SRC, "<cellvote_patch>", "exec"), mod.__dict__)
    return mod


def load_store(store: Path) -> dict[str, dict]:
    """base_key -> {sample_key: sample}, mirroring load_decoded_results."""
    out: dict[str, dict] = defaultdict(dict)
    for f in sorted(store.iterdir()):
        if not f.is_file():
            continue
        try:
            with bz2.BZ2File(f) as fh:
                samples = pickle.load(fh)
        except Exception as e:
            print(f"  skipping {f.name}: {type(e).__name__}")
            continue
        base = f.name.split(".")[0]
        for i, s in enumerate(samples):
            out[base][f"{f.name}.out{i}"] = s
    return dict(out)


def ground_truth() -> dict[str, np.ndarray]:
    """base_key -> the correct grid, from the public evaluation split."""
    truth = {}
    for t in load_split("evaluation"):
        for i, pair in enumerate(t.test):
            if pair.output is not None:
                truth[f"{t.task_id}_{i}"] = np.asarray(pair.output)
    return truth


def evaluate(decoded, truth, algo, n_guesses=2):
    """Kaggle's rule: an output counts if either of the top-2 matches exactly."""
    solved, total, misses = 0, 0, []
    for base, guesses in decoded.items():
        if base not in truth:
            continue
        total += 1
        try:
            ranked = algo(guesses)
        except Exception as e:
            misses.append((base, f"{type(e).__name__}: {e}"))
            continue
        want = truth[base]
        if any(np.array_equal(np.asarray(g), want) for g in ranked[:n_guesses]):
            solved += 1
        else:
            misses.append((base, None))
    return solved, total, misses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    ap.add_argument("--only", default="", help="run just this selector by name")
    ap.add_argument("--n-guesses", type=int, default=2)
    args = ap.parse_args()

    if not args.store.is_dir():
        raise SystemExit(
            f"no candidate store at {args.store}\n"
            "produce one with:  python agi2/repro_nvarc.py --hours 5 "
            "--full-eval --save-candidates  &&  kaggle kernels push -p agi2/out/nvarc"
        )

    dec = load_decoder(args.decoder)
    decoded = load_store(args.store)
    truth = ground_truth()

    scored = [b for b in decoded if b in truth]
    cand_counts = [len(decoded[b]) for b in scored]
    print(f"store    : {args.store}")
    print(f"outputs  : {len(scored)} with ground truth ({len(decoded)} in store)")
    print(f"candidates per output: min {min(cand_counts)}, "
          f"median {sorted(cand_counts)[len(cand_counts) // 2]}, max {max(cand_counts)}")
    print(f"top-{args.n_guesses} counts as solved\n")

    algos = list(getattr(dec, "selection_algorithms", []))
    if args.only:
        algos = [a for a in algos if a.__name__ == args.only]
        if not algos:
            raise SystemExit(f"no selector named {args.only}")

    # An oracle over the candidate pool: the best any selector could possibly do.
    def oracle(guesses):
        return [g["solution"] for g in guesses.values()]

    rows = []
    for algo in algos:
        s, n, misses = evaluate(decoded, truth, algo, args.n_guesses)
        errs = [m for m in misses if m[1]]
        rows.append((algo.__name__, s, n, len(errs)))
    s, n, _ = evaluate(decoded, truth, oracle, n_guesses=10**6)
    rows.append(("ORACLE (any candidate)", s, n, 0))

    width = max(len(r[0]) for r in rows)
    base = rows[0][1] if rows else 0
    for name, s, n, errs in rows:
        delta = "" if name == rows[0][0] else f"  ({s - base:+d} vs {rows[0][0]})"
        err = f"   [{errs} raised]" if errs else ""
        print(f"  {name:<{width}}  {s:>3}/{n}  {s / n:6.1%}{delta}{err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
