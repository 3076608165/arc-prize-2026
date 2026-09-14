"""Compare more-work variants against the augment-16 baseline on the same tasks.

Every variant ran on the same 24-task subset with its candidate store saved,
so this scores each store against ground truth with upstream's own selector
and reports, per variant: solved on the shared outputs, gained/lost vs
baseline, candidates per output, and -- from the quota it consumed -- the
per-task time that decides whether 240 tasks fit in 11.83h.

    python agi2/compare_variants.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_offline as SO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BASELINE = ("augment16 (v8)", ROOT / "agi2" / "out" / "v8")
VARIANTS = [
    ("augment24", ROOT / "agi2" / "out" / "exp-aug24" / "result"),
    ("epochs2",   ROOT / "agi2" / "out" / "exp-ep2" / "result"),
    ("infer4",    ROOT / "agi2" / "out" / "exp-inf4" / "result"),
]

#: Kaggle's per-task wall on the real rerun: 11.83h * 4 workers / 240 tasks.
FITS_MIN_PER_TASK = 11.83 * 60 * 4 / 240   # ~11.8 min


def load(d: Path):
    store = d / "inference_outputs"
    dec_py = d / "arc_decoder.py"
    if not store.is_dir() or not dec_py.exists():
        return None, None
    dec = SO.load_decoder(dec_py, with_cellvote=False)
    return dec, SO.load_store(store)


def solved_set(dec, store, truth):
    out = set()
    for base, g in store.items():
        if base not in truth:
            continue
        ranked = dec.score_full_probmul_3(g)[:2]
        if any(np.array_equal(np.asarray(x), truth[base]) for x in ranked):
            out.add(base)
    return out


def main() -> int:
    truth = SO.ground_truth()
    bdec, bstore = load(BASELINE[1])
    if bstore is None:
        raise SystemExit(f"baseline store missing at {BASELINE[1]}")
    bsolved = solved_set(bdec, bstore, truth)

    print(f"baseline {BASELINE[0]}: {len(bsolved)} solved on {len([b for b in bstore if b in truth])} outputs\n")
    print(f"{'variant':<12}{'outputs':>8}{'solved':>8}{'gained':>8}{'lost':>6}{'cands/out':>11}")
    for name, d in VARIANTS:
        dec, store = load(d)
        if store is None:
            print(f"{name:<12}   (no result yet)")
            continue
        shared = [b for b in store if b in truth and b in bstore]
        vs = solved_set(dec, store, truth) & set(shared)
        bs = bsolved & set(shared)
        cands = np.median([len(store[b]) for b in shared]) if shared else 0
        print(f"{name:<12}{len(shared):>8}{len(vs):>8}{len(vs - bs):>8}{len(bs - vs):>6}{cands:>11.0f}")
        gained = sorted(vs - bs); lost = sorted(bs - vs)
        if gained: print(f"{'':12}  + {', '.join(gained)}")
        if lost:   print(f"{'':12}  - {', '.join(lost)}")

    print(f"\nfits 240 tasks in 11.83h if per-task <= {FITS_MIN_PER_TASK:.1f} min "
          f"(read each run's quota: minutes = quota_h*60/2 for L4; per-task = that*4/24)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
