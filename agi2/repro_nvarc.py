"""Assemble a push-ready fork of the public NVARC-lineage notebook.

Why a script instead of hand-editing the .ipynb: the notebook is 38 KB of
someone else's code and we want to change exactly two things about it -- whose
account it runs under, and how long it is allowed to run. Doing that by hand is
unreviewable and undoes itself the next time we re-pull upstream.

Provenance: the pipeline is `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`
(LB 33.89), itself a minimal perf patch of the NVARC 2025 winning solution.
We are not the authors of it. The fork is pushed private and is a baseline to
measure against, not a submission we can claim as our own work.

The one thing worth understanding before spending quota: the upstream notebook
sets a 12-hour budget and runs on L4 x4, which consumes GPU quota at 2x. A
single full run costs roughly 24 hours against a weekly allowance of about 30.
So validate the plumbing with `--hours 1` first; only then spend a full run.

    python agi2/repro_nvarc.py --hours 1        # cheap plumbing check
    python agi2/repro_nvarc.py --hours 11.83    # the real thing
    kaggle kernels push -p agi2/out/nvarc
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "mikelou1/arc-agi2-lb33-89-minimal-perfpatch"
CACHE = ROOT / "agi2" / "out" / "refs" / "arc-agi2-lb33-89-minimal-perfpatch"

OUR_ID = "uts3125025522741/arc-agi-2-nvarc-repro"
OUR_TITLE = "ARC AGI 2 NVARC repro"     # must slugify to the id, or push 409s

ATTRIBUTION = f"""\
# NVARC repro (fork)

Fork of [`{UPSTREAM}`](https://www.kaggle.com/code/{UPSTREAM})
(LB 33.89), itself a minimal performance patch of the NVARC ARC Prize 2025
winning solution. **Not our work** -- run here as a baseline to measure our own
methods against.

Local experiment settings are recorded in `experiment.json` beside this notebook.
The model and core solver are inherited from the attributed upstream work.
"""


def fetch_upstream(force: bool = False) -> Path:
    nb = next(iter(CACHE.glob("*.ipynb")), None)
    if nb is not None and not force:
        return nb
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "pull", UPSTREAM,
         "-p", str(CACHE), "-m"],
        check=True, capture_output=True, text=True,
    )
    nb = next(iter(CACHE.glob("*.ipynb")), None)
    if nb is None:
        raise SystemExit(f"pull of {UPSTREAM} produced no notebook")
    return nb


def cell_text(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


#: Upstream's four-task shortcut, verbatim, inside `starter.py`. The whole
#: `if not rerun_mode:` block has to go -- dropping only the inner two lines
#: leaves a header with no body, i.e. an IndentationError and a wasted run.
ALLOWLIST = (
    "        if not rerun_mode:\n"
    '            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:\n'
    "                continue\n"
)


def drop_allowlist(nb: dict) -> bool:
    """Remove the four-task shortcut so a commit run covers all 120 eval tasks.

    `starter.py` queues only four hardcoded ids whenever
    KAGGLE_IS_COMPETITION_RERUN is unset, to keep notebook commits cheap. That
    also means a commit run measures nothing about speed. Dropping it turns a
    cheap 1h commit into a throughput probe: queue all 120 evaluation tasks,
    run the clock out, then count how many got a real prediction. That is the
    number the 12h full-run decision actually depends on, and it has to be
    measured on the 4x L4 the submission will run on -- not on other hardware.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if ALLOWLIST in txt:
            cell["source"] = txt.replace(ALLOWLIST, "").splitlines(keepends=True)
            return True
    return False


def set_subset(nb: dict, n: int) -> list[str]:
    """Replace the four-task allowlist with the first `n` evaluation task ids.

    Turns a commit run into a controlled compute experiment: hold the task set
    fixed, change only the per-task budget, and compare against the same tasks
    from an earlier run. Upstream's allowlist mechanism is already the right
    hook -- it just needs a longer list.
    """
    data = ROOT / "data" / "ARC-AGI-2" / "data" / "evaluation"
    ids = sorted(p.stem for p in data.glob("*.json"))[:n]
    replacement = (
        "        if not rerun_mode:\n"
        f"            if key not in {ids!r}:\n"
        "                continue\n"
    )
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if ALLOWLIST in txt:
            cell["source"] = txt.replace(ALLOWLIST, replacement).splitlines(
                keepends=True)
            return ids
    return []


def set_train_augment(nb: dict, n: int) -> bool:
    """Change the test-time-training augmentation count in arc_solver.py.

    `train_ds = puzzle_ds.augment(n=16, ...)` is the dominant per-task cost:
    16 augmented views of the demonstrations, one LoRA epoch over them, per
    puzzle. Per-task work is fixed rather than elastic (see findings), so this
    is the actual knob for trading quality against how many of the 240 tasks
    finish inside the 12h wall -- which we measured at a 50-minute margin.
    """
    import re as _re
    pat = _re.compile(r"(train_ds = puzzle_ds\.augment\(n=)\d+")
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if pat.search(txt):
            cell["source"] = pat.sub(rf"\g<1>{n}", txt).splitlines(keepends=True)
            return True
    return False


def add_cellvote(nb: dict) -> bool:
    """Append our per-cell voting selector to the arc_decoder.py cell.

    It registers itself in `selection_algorithms`, so a commit run's
    `benchmark_selection_algos()` prints its accuracy next to upstream's two on
    the same decoded candidates -- an ablation that costs no extra compute.
    """
    sys.path.insert(0, str(ROOT / "agi2"))
    from cellvote_patch import CELLVOTE_SRC

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if "%%writefile arc_decoder.py" in txt:
            cell["source"] = (txt.rstrip() + "\n" + CELLVOTE_SRC).splitlines(
                keepends=True)
            return True
    return False


#: Copies the decoded-candidate store into the kernel's output. One run then
#: yields a permanent local corpus on which any number of selection algorithms
#: can be compared offline, at zero further quota -- instead of one 10h run per
#: selector. The store is what `ArcDecoder.load_decoded_results` reads.
SAVE_CANDIDATES_SRC = """

# --- save the candidate store (added by agi2/repro_nvarc.py) ----------------
import shutil as _shutil, os as _os
_src, _dst = "/kaggle/inference_outputs", "/kaggle/working/inference_outputs"
if _os.path.isdir(_src):
    _shutil.rmtree(_dst, ignore_errors=True)
    _shutil.copytree(_src, _dst)
    _n = sum(len(f) for _, _, f in _os.walk(_dst))
    _mb = sum(_os.path.getsize(_os.path.join(r, f))
              for r, _, fs in _os.walk(_dst) for f in fs) / 1e6
    print(f"saved candidate store: {_n} files, {_mb:.1f} MB")
else:
    print(f"NO CANDIDATE STORE at {_src}")
"""


def save_candidates(nb: dict) -> bool:
    """Append the store copy to the final cell, after the submission is written."""
    for cell in reversed(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if "submission.json" in txt and "json.dump" in txt:
            cell["source"] = (txt.rstrip() + "\n" + SAVE_CANDIDATES_SRC).splitlines(
                keepends=True)
            return True
    return False


def set_budget(nb: dict, hours: float) -> bool:
    """Rewrite the single `global_end_time = ...` line."""
    pat = re.compile(r"global_end_time\s*=\s*time\.time\(\)\s*\+[^\n]*")
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if pat.search(txt):
            secs = int(hours * 3600)
            cell["source"] = pat.sub(
                f"global_end_time = time.time() + {secs}  # {hours}h budget "
                f"(upstream: 12h minus a 10 min write buffer)",
                txt,
            ).splitlines(keepends=True)
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=1.0,
                    help="wall-clock budget. Upstream uses 11.83 (12h - 10min).")
    ap.add_argument("--out", default="agi2/out/nvarc")
    ap.add_argument("--refresh", action="store_true", help="re-pull upstream")
    ap.add_argument("--save-candidates", action="store_true",
                    help="copy the decoded-candidate store into kernel output so "
                         "selection algorithms can be compared offline")
    ap.add_argument("--cellvote", action="store_true",
                    help="add our per-cell voting selector as an extra arm in "
                         "benchmark_selection_algos()")
    ap.add_argument("--augment", type=int, default=0,
                    help="test-time-training augmentation count (upstream: 16). "
                         "The dominant per-task cost.")
    ap.add_argument("--subset", type=int, default=0,
                    help="commit-mode: queue the first N evaluation tasks instead "
                         "of upstream's 4, for controlled compute experiments")
    ap.add_argument("--full-eval", action="store_true",
                    help="queue all 120 evaluation tasks in commit mode instead of "
                         "upstream's 4-task shortcut, to measure throughput")
    args = ap.parse_args()
    if not 0 < args.hours <= 11.83:
        ap.error("--hours must be positive and at most 11.83")
    if args.augment < 0 or args.subset < 0:
        ap.error("--augment and --subset cannot be negative")
    if args.subset and args.full_eval:
        ap.error("--subset and --full-eval are mutually exclusive")

    src_nb = fetch_upstream(force=args.refresh)
    nb = json.loads(src_nb.read_text(encoding="utf-8"))
    up_meta = json.loads((CACHE / "kernel-metadata.json").read_text(encoding="utf-8"))

    if not set_budget(nb, args.hours):
        raise SystemExit("could not find `global_end_time = time.time() + ...`; "
                         "upstream changed shape, re-read it before pushing")

    if args.save_candidates and not save_candidates(nb):
        raise SystemExit("could not find the submission-writing cell; upstream "
                         "changed shape, re-read it before pushing")

    if args.cellvote and not add_cellvote(nb):
        raise SystemExit("could not find the arc_decoder.py cell; upstream "
                         "changed shape, re-read it before pushing")

    if args.augment and not set_train_augment(nb, args.augment):
        raise SystemExit("could not find `train_ds = puzzle_ds.augment(n=...)`; "
                         "upstream changed shape, re-read it before pushing")

    subset_ids = []
    if args.subset:
        subset_ids = set_subset(nb, args.subset)
        if not subset_ids:
            raise SystemExit("could not find the allowlist to replace")

    if args.full_eval and not drop_allowlist(nb):
        raise SystemExit("could not find the four-task allowlist in starter.py; "
                         "upstream changed shape, re-read it before pushing")

    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {},
                           "source": ATTRIBUTION.splitlines(keepends=True)})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # Carry every source the upstream notebook depends on. Dropping any of
    # these produces a run that fails minutes in, having already spent quota.
    meta = {
        "id": OUR_ID,
        "title": OUR_TITLE,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": up_meta.get("dataset_sources", []),
        "kernel_sources": up_meta.get("kernel_sources", []),
        "competition_sources": up_meta.get("competition_sources", []),
        "model_sources": up_meta.get("model_sources", []),
    }
    if up_meta.get("machine_shape"):
        meta["machine_shape"] = up_meta["machine_shape"]
    # Carrying this is not optional. The unsloth/torch/numpy wheels in the
    # `sorokin/pip-install-unsloth-flash-patch` utility kernel are built against
    # a pinned BYOD image. Push without it and Kaggle runs the current default
    # image instead, where those wheels fail to load their C extensions:
    #   ImportError: Failed to load PyTorch C extensions
    #   ModuleNotFoundError: No module named 'numpy._core._multiarray_umath'
    if up_meta.get("docker_image"):
        meta["docker_image"] = up_meta["docker_image"]
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                              encoding="utf-8")
    (out / "experiment.json").write_text(json.dumps({
        "upstream": UPSTREAM, "hours": args.hours,
        "train_augment": args.augment or 16,
        "commit_subset": subset_ids or ("all" if args.full_eval else "upstream-four"),
        "competition_tasks": "all", "save_candidates": args.save_candidates,
        "cellvote": args.cellvote,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"upstream      : {UPSTREAM}")
    queued = "all 120 eval tasks" if args.full_eval else "upstream 4-task shortcut"
    print(f"budget        : {args.hours}h")
    if subset_ids:
        queued = f"{len(subset_ids)} tasks (subset)"
    print(f"commit mode   : {queued}")
    print(f"train augment : {args.augment or 16} (upstream 16)")
    print(f"cell-vote arm : {'added' if args.cellvote else 'no'}")
    print(f"accelerator   : {meta.get('machine_shape', 'default GPU')}"
          f"   (L4 burns GPU quota at 2x)")
    print(f"model         : {', '.join(meta['model_sources']) or 'none'}")
    print(f"kernel deps   : {', '.join(meta['kernel_sources']) or 'none'}")
    print(f"\nwrote {out}\n\npush with:\n  kaggle kernels push -p {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
