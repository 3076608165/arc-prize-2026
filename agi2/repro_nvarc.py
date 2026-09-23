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

The knob that changes capability rather than cost is `--base-model`: the whole
family is per-task LoRA test-time training on a grid-SFT checkpoint, and the
attached one is a 4B. It is only a one-line swap if the candidate carries the
same 16-token ARC vocab -- see `set_base_model()`.

    python agi2/repro_nvarc.py --hours 4 --subset 24 --save-candidates \
        --out agi2/out/exp-m8b \
        --kernel-id uts3125025522741/arc-agi-2-exp-m8b --title "ARC AGI 2 exp m8b" \
        --base-model owner/slug/Transformers/default/1
"""
from __future__ import annotations

import argparse
import hashlib
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


def set_epochs(nb: dict, n: int) -> bool:
    """num_train_epochs in arc_solver.py. Upstream: 1."""
    import re as _re
    pat=_re.compile(r"(num_train_epochs=)\d+")
    for cell in nb["cells"]:
        if cell["cell_type"]!="code": continue
        txt=cell_text(cell)
        if pat.search(txt):
            cell["source"]=pat.sub(rf"\g<1>{n}",txt).splitlines(keepends=True); return True
    return False


def set_infer_augment(nb: dict, n: int) -> bool:
    """eval_ds = puzzle_ds_multi.augment(n=...) -- inference views. Upstream: 2."""
    import re as _re
    pat=_re.compile(r"(eval_ds = puzzle_ds_multi\.augment\(n=)\d+")
    for cell in nb["cells"]:
        if cell["cell_type"]!="code": continue
        txt=cell_text(cell)
        if pat.search(txt):
            cell["source"]=pat.sub(rf"\g<1>{n}",txt).splitlines(keepends=True); return True
    return False


MODEL_MOUNT = "/kaggle/input/models"


def mount_path(ref: str) -> str:
    """Kaggle model ref -> the path the *kernel* sees.

    `model_sources` is written `<owner>/<slug>/<Framework>/<instance>/<version>`
    and the mount lowercases the framework segment and nothing else. Verified
    against upstream, which attaches
    `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` and reads
    `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`.
    """
    owner, slug, framework, instance, version = ref.split("/")
    return f"{MODEL_MOUNT}/{owner}/{slug}/{framework.lower()}/{instance}/{version}"


def set_base_model(nb: dict, ref: str) -> bool:
    """Point the pipeline at a different attached base model.

    The whole family is per-task LoRA test-time training on top of a grid-SFT
    checkpoint, and the attached one is a 4B. A larger checkpoint with the
    *same custom tokenizer* is a one-line swap. One with a stock Qwen tokenizer
    is not: `turbo_dfs` restricts the search to the 16 ids in `ARC_VOCAB`
    (0-9 plus separators, USER_TOKEN_ID=11, ASSISTANT_TOKEN_ID=12, PAD_ID=13,
    EOS_ID=15) and the collator masks on those ids. A wrong tokenizer does not
    crash -- it trains and scores garbage, which costs a full run to discover.
    """
    pat = re.compile(r'model_name="[^"]*"')
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if pat.search(txt):
            cell["source"] = pat.sub(f'model_name="{mount_path(ref)}"',
                                     txt).splitlines(keepends=True)
            return True
    return False


def add_model_probe(nb: dict) -> bool:
    """Log the swapped checkpoint's tokenizer before training starts.

    `turbo_dfs` restricts the search to the 16 ids in `ARC_VOCAB` and the
    collator masks on USER_TOKEN_ID / ASSISTANT_TOKEN_ID / EOS_ID. A checkpoint
    with a different tokenizer does not necessarily fail loudly -- it can train,
    decode, and score something that is not a grid answer. Two lines of log turn
    a wasted 4-hour run into a two-minute one.
    """
    anchor = "    model = FastLanguageModel.get_peft_model(model, **peft_params)"
    probe = (
        "\n    # --- base-model probe (added by agi2/repro_nvarc.py) ----------------\n"
        '    print(f"base model  : {model_name}", flush=True)\n'
        '    print(f"vocab size  : {getattr(tokenizer, \'vocab_size\', \'?\')}", flush=True)\n'
        '    print(f"ARC_TOKENS  : {tokenizer.decode(ARC_TOKENS)!r}", flush=True)\n'
    )
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if txt.count(anchor) == 1:
            cell["source"] = txt.replace(anchor, probe + "\n" + anchor).splitlines(
                keepends=True)
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


#: Second-pass machinery. The pipeline is one pinned notebook over per-task
#: LoRA test-time training, and byte-identical reruns of it land anywhere in
#: 28.9-31.0 -- that variance is a decorrelated redraw we currently only get by
#: spending another day's slot. A second pass with different augmentation seeds
#: turns part of it into candidates inside ONE submission.
PASS2_MARKER = "__PASS2__"
PASS2_STORE = "/kaggle/inference_outputs_p2"
PASS2_QUEUE_ANCHOR = "    for key in sorted(data.keys()):"
PASS2_GET_ANCHOR = ("        key = queue.get()\n"
                    "        if key is None:\n"
                    "            break\n")
PASS2_TRAIN_SEED = "train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)"
PASS2_EVAL_SEED = "eval_ds = puzzle_ds_multi.augment(n=2, seed=2)"


def _requeue_pass(nb: dict) -> bool:
    """Queue every task a second time, behind a marker the worker understands.

    Pass B goes *behind* all of pass A, so it runs only on slack: a task costs
    ~554 s (measured, 1 h full-eval run, 24 tasks finished) and 120 tasks on 4
    workers leaves room for ~2.5 passes inside the 11.83 h wall. If pass A runs
    long, pass B simply never starts -- the run degrades to today's behaviour.

    The filter lines (the commit-mode allowlist, or the subset ids, or whatever
    --full-eval left) are copied verbatim into the second loop, so the two
    passes always cover the same task set.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if PASS2_QUEUE_ANCHOR not in txt:
            continue
        end = txt.index("        queue.put(key)\n", txt.index(PASS2_QUEUE_ANCHOR))
        body = txt[txt.index(PASS2_QUEUE_ANCHOR) + len(PASS2_QUEUE_ANCHOR):end]
        head = txt[:txt.index(PASS2_QUEUE_ANCHOR)]
        new = (
            "    _queue_keys = sorted(data.keys())\n"
            "    for key in _queue_keys:" + body + "        queue.put(key)\n"
            f'    queue.put("{PASS2_MARKER}")\n'
            "    for key in _queue_keys:" + body + "        queue.put(key)\n"
        )
        cell["source"] = (head + new + txt[end + len("        queue.put(key)\n"):]
                          ).splitlines(keepends=True)
        return True
    return False


def _split_passes(nb: dict) -> bool:
    """Worker side: pass B gets its own store and its own augmentation seeds."""
    done = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if 'dir_outputs = "/kaggle/inference_outputs"' not in txt:
            continue
        if PASS2_GET_ANCHOR not in txt:
            continue
        txt = txt.replace(
            PASS2_GET_ANCHOR,
            PASS2_GET_ANCHOR +
            f'        if key == "{PASS2_MARKER}":\n'
            "            pass_id = 1\n"
            f'            dir_outputs = "{PASS2_STORE}"\n'
            "            os.makedirs(dir_outputs, exist_ok=True)\n"
            '            print(f"[Rank {rank}] second pass: decorrelated TTT", flush=True)\n'
            "            continue\n", 1)
        txt = txt.replace('    dir_outputs = "/kaggle/inference_outputs"',
                          '    pass_id = 0\n'
                          '    dir_outputs = "/kaggle/inference_outputs"', 1)
        #: 101/103 are arbitrary odd offsets; the only requirement is that
        #: `augment(seed=...)` -- which seeds numpy globally -- sees a different
        #: value, so pass B's permutes and demo orderings differ from pass A's.
        txt = txt.replace(PASS2_TRAIN_SEED,
                          PASS2_TRAIN_SEED.replace("seed=1", "seed=1 + 101 * pass_id"), 1)
        txt = txt.replace(PASS2_EVAL_SEED,
                          PASS2_EVAL_SEED.replace("seed=2", "seed=2 + 103 * pass_id"), 1)
        cell["source"] = txt.splitlines(keepends=True)
        done += 1
    return done == 1


def _pool_selection(nb: dict, hours_note: str = "") -> bool:
    """Submission side: pool both stores, keep pass A's top-1 as attempt_1.

    Monotone by construction. `score_kgmon` picks a top-2 from the pooled
    candidates, but attempt_1 is forced to pass A's own top-1, exactly the grid
    today's submission would have started with -- so a pass B that is noisier
    than pass A can only change attempt_2. Without this the second pass is a
    second coin flip on the first attempt as well, and a bad draw costs more
    than it can gain.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = cell_text(cell)
        if 'decoder.load_decoded_results("/kaggle/inference_outputs")\n' not in txt:
            continue
        txt = txt.replace(
            "from arc_decoder import ArcDecoder",
            "from arc_decoder import ArcDecoder, hashable, score_kgmon", 1)
        txt = txt.replace(
            'decoder.load_decoded_results("/kaggle/inference_outputs")\n',
            'decoder.load_decoded_results("/kaggle/inference_outputs")\n'
            f'_pooled = os.path.isdir("{PASS2_STORE}")\n'
            "if _pooled:\n"
            f'    decoder.load_decoded_results("{PASS2_STORE}", run_name=".p2")\n'
            "    print(f\"[two-pass] pooled {len(decoder.decoded_results)} outputs "
            "from both passes\")\n", 1)
        txt = txt.replace(
            "submission = data.get_submission(decoder.run_selection_algo())\n",
            "selected = decoder.run_selection_algo(score_kgmon)\n"
            "if _pooled:\n"
            "    _pass_a = ArcDecoder(data.split_multi_replies(), n_guesses=2)\n"
            '    _pass_a.load_decoded_results("/kaggle/inference_outputs")\n'
            "    _a_only = _pass_a.run_selection_algo(score_kgmon)\n"
            "    _merged = {}\n"
            "    for _bk, _cands in selected.items():\n"
            "        _top = _a_only.get(_bk) or []\n"
            "        _first = _top[0] if _top else None\n"
            "        _seen = {hashable(_first['solution'])} if _first else set()\n"
            "        _merged[_bk] = ([_first] if _first else []) + [\n"
            "            c for c in _cands if hashable(c['solution']) not in _seen]\n"
            "    selected = _merged\n"
            "    print(\"[two-pass] attempt_1 pinned to pass A's top-1\")\n"
            "submission = data.get_submission(selected)\n", 1)
        cell["source"] = txt.splitlines(keepends=True)
        return True
    return False


def add_two_pass_pooling(nb: dict) -> bool:
    """All three edits, or none -- a half-applied second pass wastes a run."""
    return _requeue_pass(nb) and _split_passes(nb) and _pool_selection(nb)


LARGEST_FIRST_ANCHOR = "    _queue_keys = sorted(data.keys())\n"

#: Cost proxy for queue ordering: total input cells, computed in the starter
#: process where the whole task set is already in memory.
LARGEST_FIRST_KEYS = """    # Largest puzzles first. The four workers otherwise reach the expensive tail
    # last, and the tail is exactly what a second pass never gets to. Proxy is
    # total input cells; madarshbb (discussion 729743) measures this ordering as
    # a speed win on this same pipeline.
    _queue_keys = sorted(
        data.keys(),
        key=lambda k: -sum(len(g) * len(g[0]) for t in data[k]["test"] for g in (t["input"],)))


"""


def order_largest_first(nb: dict) -> bool:
    """Queue the expensive tasks first, so the cheap ones fill the tail.

    Pure scheduling: the candidate pool is unchanged, only which tasks are
    attempted when. Applied after add_two_pass_pooling, whose key list it
    replaces.
    """
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        txt = "".join(cell["source"])
        if txt.count(LARGEST_FIRST_ANCHOR) != 1:
            continue
        cell["source"] = (txt.replace(LARGEST_FIRST_ANCHOR, LARGEST_FIRST_KEYS)).splitlines(keepends=True)
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
    ap.add_argument("--kernel-id", default=OUR_ID)
    ap.add_argument("--title", default=OUR_TITLE)
    ap.add_argument("--worker-hash-seed", type=int, default=None,
                    help="set PYTHONHASHSEED for subsequently launched workers")
    ap.add_argument("--refresh", action="store_true", help="re-pull upstream")
    ap.add_argument("--audit-coverage", action="store_true", help="record per-task stage and candidate coverage events")
    ap.add_argument("--save-candidates", action="store_true",
                    help="copy the decoded-candidate store into kernel output so "
                         "selection algorithms can be compared offline")
    ap.add_argument("--cellvote", action="store_true",
                    help="add our per-cell voting selector as an extra arm in "
                         "benchmark_selection_algos()")
    ap.add_argument("--augment", type=int, default=0,
                    help="test-time-training augmentation count (upstream: 16). "
                         "The dominant per-task cost.")
    ap.add_argument("--epochs", type=int, default=0, help="num_train_epochs (upstream 1)")
    ap.add_argument("--infer-augment", type=int, default=0, help="inference views (upstream 2)")
    ap.add_argument("--base-model", default="",
                    help="swap the attached base checkpoint, as a Kaggle model ref "
                         "`<owner>/<slug>/<Framework>/<instance>/<version>` (upstream: "
                         "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1). "
                         "Only safe if the checkpoint carries the same 16-token "
                         "ARC vocab -- see set_base_model().")
    ap.add_argument("--subset", type=int, default=0,
                    help="commit-mode: queue the first N evaluation tasks instead "
                         "of upstream's 4, for controlled compute experiments")
    ap.add_argument("--full-eval", action="store_true",
                    help="queue all 120 evaluation tasks in commit mode instead of "
                         "upstream's 4-task shortcut, to measure throughput")
    ap.add_argument("--two-pass", action="store_true",
                    help="run a second, decorrelated TTT pass behind pass A on the "
                         "same task set, pool both candidate stores, and pin "
                         "attempt_1 to pass A's top-1 (monotone).")
    ap.add_argument("--largest-first", action="store_true",
                    help="order the shared task queue by descending input size "
                         "(needs --two-pass); scheduling only, pool unchanged")
    args = ap.parse_args()
    if not 0 < args.hours <= 11.83:
        ap.error("--hours must be positive and at most 11.83")
    if args.augment < 0 or args.subset < 0:
        ap.error("--augment and --subset cannot be negative")
    if args.subset and args.full_eval:
        ap.error("--subset and --full-eval are mutually exclusive")
    if args.worker_hash_seed is not None and not 0 <= args.worker_hash_seed <= 4294967295:
        ap.error("--worker-hash-seed must be in [0, 4294967295]")
    if args.base_model and len(args.base_model.split("/")) != 5:
        ap.error("--base-model must be `<owner>/<slug>/<Framework>/<instance>/<version>`")

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

    if args.epochs and not set_epochs(nb, args.epochs):
        raise SystemExit("could not find num_train_epochs")
    if args.infer_augment and not set_infer_augment(nb, args.infer_augment):
        raise SystemExit("could not find eval_ds augment")
    if args.base_model and not set_base_model(nb, args.base_model):
        raise SystemExit("could not find `model_name=`; upstream changed shape")
    if args.base_model and not add_model_probe(nb):
        raise SystemExit("could not find the get_peft_model anchor for the model probe")

    subset_ids = []
    if args.subset:
        subset_ids = set_subset(nb, args.subset)
        if not subset_ids:
            raise SystemExit("could not find the allowlist to replace")

    if args.full_eval and not drop_allowlist(nb):
        raise SystemExit("could not find the four-task allowlist in starter.py; "
                         "upstream changed shape, re-read it before pushing")

    # After the task-set patches, and after coverage_audit (whose anchors are
    # single-occurrence): the second pass copies whatever filter and audit lines
    # --full-eval / --subset / --audit-coverage left behind.
    if args.audit_coverage:
        from coverage_audit import patch_notebook
        patch_notebook(nb)

    if args.two_pass and not add_two_pass_pooling(nb):
        raise SystemExit("could not apply the two-pass pooling patches; upstream "
                         "changed shape, re-read it before pushing")

    if args.largest_first:
        if not args.two_pass:
            raise SystemExit("--largest-first reorders add_two_pass_pooling's key "
                             "list; pass --two-pass with it")
        if not order_largest_first(nb):
            raise SystemExit("could not reorder the queue; upstream changed shape")

    nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {},
                           "source": ATTRIBUTION.splitlines(keepends=True)})
    if args.worker_hash_seed is not None:
        nb["cells"].insert(1, {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None, "source": ["import os\n",
                f'os.environ["PYTHONHASHSEED"] = "{args.worker_hash_seed}"\n']})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "submission.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")

    # Carry every source the upstream notebook depends on. Dropping any of
    # these produces a run that fails minutes in, having already spent quota.
    meta = {
        "id": args.kernel_id,
        "title": args.title,
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
        "model_sources": ([args.base_model] if args.base_model
                          else up_meta.get("model_sources", [])),
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
        "infer_augment": args.infer_augment or 2,
        "epochs": args.epochs or 1,
        "base_model": args.base_model or (up_meta.get("model_sources") or [""])[0],
        "commit_subset": subset_ids or ("all" if args.full_eval else "upstream-four"),
        "competition_tasks": "all", "save_candidates": args.save_candidates,
        "cellvote": args.cellvote,
        "two_pass": args.two_pass,
        "largest_first": args.largest_first,
        "audit_coverage": args.audit_coverage,
        "worker_hash_seed": args.worker_hash_seed,
        "kernel_id": args.kernel_id,
        "upstream_notebook_sha256": hashlib.sha256(src_nb.read_bytes()).hexdigest(),
        "notebook_sha256": hashlib.sha256((out / "submission.ipynb").read_bytes()).hexdigest(),
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
