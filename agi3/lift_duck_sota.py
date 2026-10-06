#!/usr/bin/env python3
"""Lift the best open-source ARC-AGI-3 solution into our own kernel.

The public record is unambiguous on this board: `dfranzen/arc-agi-3-milestone-2-solution`
(351 votes, the top-voted ARC-AGI-3 notebook) is Tufa Labs' Duck harness plus Daniel
Franzen's modifications, fully offline, and the community reads it at ~27.8 LB. Our own
best pair on this board is 0.67 + 0.27.

This script does NOT rewrite the solution. It copies the published notebook verbatim and
makes exactly ONE change, in the commit-mode branch only:

    the non-submission (`TRUE_SUBMISSION == False`) branch is the commit run, whose only
    job is to produce `submission.parquet` so the version becomes submittable. Upstream
    sets `n_passes = 4` over all 25 demo games there, which spends a second 9 h GPU
    session per slot for nothing: the score comes from the competition rerun, which takes
    the `TRUE_SUBMISSION` branch and is untouched by this patch.

So the lift is a commit-mode allowlist (the campaign's standing pattern), not a
behavioural variant. Everything else - the harness patch, the launcher, the solver
config, the model, the pinned docker digest - is byte-identical to upstream.

Provenance: the source notebook sha256 and our one diff are both recorded in
experiment.json next to the build.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "agi3" / "out" / "duck-sota"
SRC_DIR = OUT / "refs" / "upstream"

# The lift plus, in the second variant, exactly one more knob: the launcher's serving
# batch. Upstream serves `--max-running-requests 10` while the scored rerun drives
# `bm.solver.concurrency = 120` over ~110 private games, so at most 10 games hold a
# decode slot at any instant. The commit smoke measured the consequence on the public
# 25: 146 actions, 133,521 tokens, 556 tok/s aggregate, and 13 of 25 games served ZERO
# actions inside their 240 s budget. Decode on an INT4 hybrid is weight-read-bound, so
# batch is the lever the config is leaving unspent -- and it is measurable for free in
# the same 4-minute smoke, before the arm is ever scored.
VARIANTS = {
    "v1": {"out": "duck-sota", "cfgs": {}, "envs": {},
           "note": "verbatim lift; the only diff is the bounded commit-mode smoke"},
    "batch32": {
        "out": "duck-sota-v2",
        "cfgs": {"MAXREQ": 32, "CUDAGRAPH_MAXBS": 32},
        "envs": {},
        "note": ("serving batch 10 -> 32 via MAXREQ + CUDAGRAPH_MAXBS together (the "
                 "launcher asserts CUDAGRAPH_MAXBS >= MAXREQ, so they move as one "
                 "knob). MAMBA_CACHE stays 60 > 32, so every decode slot still has a "
                 "resident Mamba/SSM state slot: the change adds KV pressure only."),
    },
    # 2026-10-06: batch32's own smoke refuted its premise -- MAXREQ 32 was capped to 12
    # by the mamba state cache (5 slots/request x 60 slots; SGLang said so in serve.log)
    # and the harness's priority gate admits 10 streams regardless. With aggregate
    # throughput flat at ~510-556 tok/s and a fixed wall clock, tokens-per-game is
    # aggregate x wall / n_games, i.e. independent of concurrency. The only lever left
    # is tokens spent per action: completion is reasoning + a ~50-token tool call, so
    # this variant turns the reasoning off and reads what it was worth.
    "lean": {
        "out": "duck-sota-v3",
        "cfgs": {},
        "envs": {"LOCAL_ANALYZER_ENABLE_THINKING": "false"},
        # MEASURED 2026-10-06 (commit smoke, kernel v3): 183 completion tokens/request (from 1,132),
    # 693 actions (from 158, 4.4x), public-25 mean score 0.0041 (from 0.4589). The actions rose
    # exactly as the arithmetic said and the score collapsed to 0.9% of v1, so reasoning is the
    # product, not the overhead: this variant is a why-parameterisation, never a submission.
    "note": ("why-parameterisation of the agent's token spend: "
                 "LOCAL_ANALYZER_ENABLE_THINKING true -> false. The smoke attributes "
                 "141,548 completion tokens over 125 requests (mean 1,132; p90 2,315; "
                 "max 5,023) and the completion is reasoning plus a one-token action, "
                 "so this knob is the whole per-action token cost. Both branches see "
                 "the same env dict, so the commit smoke measures the same setting the "
                 "scored rerun would use."),
    },
}

SOURCE_KERNEL = "dfranzen/arc-agi-3-milestone-2-solution"
# The slug Kaggle actually created for our copy: the kernel id must equal this or the next
# push creates a second kernel instead of version 2 (measured 2026-10-04 -- a title whose
# slug differs from the id makes `kernels push` file a new notebook).
OUR_KERNEL = "uts3125025522741/arc-agi-3-duck-sota-open-source-lift"
OUR_TITLE = "ARC AGI 3 duck sota (open-source lift)"
DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:"
    "57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c"
)
DATASET_SOURCES = [
    "dfranzen/pennyroyal-v253",
    "dfranzen/taaf-kaggle-source-bundle-copy",
]
MODEL_SOURCES = [
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1",
    "dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1",
]

# The one diff. Anchored on the non-submission tail of the customization-hook cell, which
# contains the demo-run concurrency arithmetic and the 32400 s budget assertion.
COMMIT_MODE_OLD = """    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = (25 - len(demo_excluded_games)) * bm.n_passes
        bm.solver.max_runtime_s_per_game = 532*60 * bm.solver.concurrency // 110
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        raise RuntimeError(
            f'Expected the 32400-second notebook budget, got {target.max_runtime_s!r}.'
        )"""

COMMIT_MODE_NEW = """    # --- commit-mode smoke (our only diff from upstream) ---------------------------
    # This branch runs only outside a competition rerun, i.e. when Kaggle commits the
    # notebook. Its single job is to leave a submission.parquet so the version becomes
    # submittable; the score comes from the rerun, which takes the TRUE_SUBMISSION
    # branch above and is unaffected by anything here. Upstream spends a full 4-pass
    # demo sweep (~9 h of GPU) in this branch, which buys nothing.
    bm.n_passes = 1
    if USE_PRIORITY_SCHEDULING:
        bm.solver.concurrency = 25
        bm.solver.max_runtime_s_per_game = 240.0
    if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
        print(
            f'[commit-smoke] WARNING unexpected notebook budget '
            f'{target.max_runtime_s!r} (expected 32400.0); continuing anyway'
        )
    # ------------------------------------------------------------------------------"""
# NOTE: the file's true last line is "    # -----" (one short rule) -- keep the marker
# text stable so the self-test can find the patch by grep.

MARKER = "commit-mode smoke (our only diff from upstream)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cells_source(nb: dict) -> list[str]:
    return ["".join(c.get("source", [])) for c in nb["cells"]]


def pull() -> Path:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    ipynb = SRC_DIR / f"{SOURCE_KERNEL.split('/')[-1]}.ipynb"
    if not ipynb.is_file():
        subprocess.run(
            ["kaggle", "kernels", "pull", SOURCE_KERNEL, "-p", str(SRC_DIR), "-m"],
            check=True,
        )
    if not ipynb.is_file():
        raise SystemExit(f"pull produced no notebook at {ipynb}")
    return ipynb


def build(variant: str = "v1") -> dict:
    global OUT, SRC_DIR
    spec = VARIANTS[variant]
    OUT = REPO / "agi3" / "out" / spec["out"]
    SRC_DIR = OUT / "refs" / "upstream"
    src = pull()
    src_sha = sha256(src)
    nb = json.loads(src.read_text(encoding="utf-8"))

    # --- the one anchored replacement, exactly once, in exactly one cell ------------
    hits = [
        i
        for i, c in enumerate(nb["cells"])
        if c.get("cell_type") == "code" and COMMIT_MODE_OLD in "".join(c.get("source", []))
    ]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 cell carrying the commit-mode branch, got {hits}")
    idx = hits[0]
    body = "".join(nb["cells"][idx]["source"])
    nb["cells"][idx]["source"] = body.replace(COMMIT_MODE_OLD, COMMIT_MODE_NEW).splitlines(
        keepends=True
    )
    if MARKER not in "".join(nb["cells"][idx]["source"]):
        raise SystemExit("patch did not land")

    # --- variant knobs: the launcher's CFG dict (cell 12) ---------------------------
    if spec["cfgs"]:
        cfg_old = "".join(f"    {k}=10,\n" for k in spec["cfgs"])
        cfg_new = "".join(f"    {k}={v},\n" for k, v in spec["cfgs"].items())
        hits = [
            i
            for i, c in enumerate(nb["cells"])
            if c.get("cell_type") == "code" and cfg_old in "".join(c.get("source", []))
        ]
        if len(hits) != 1:
            raise SystemExit(f"expected exactly 1 cell carrying {cfg_old!r}, got {hits}")
        kn = hits[0]
        body = "".join(nb["cells"][kn]["source"]).replace(cfg_old, cfg_new)
        nb["cells"][kn]["source"] = body.splitlines(keepends=True)
        if cfg_new not in body or cfg_old in body:
            raise SystemExit("serving knob did not land")
        # The knob is useless if it does not reach the scored rerun's branch, and the
        # bounds check in the launcher would refuse to boot a config where the CUDA-graph
        # batch does not cover MAXREQ.
        if f'max-running-requests", str(CFG["MAXREQ"])' not in body:
            raise SystemExit("serving knob landed in a cell that never builds the SGLang args")
        assert all(v >= 10 for v in spec["cfgs"].values())

    # --- variant knobs: the harness env dict (cell 4) -------------------------------
    if spec.get("envs"):
        hits = []
        for i, c in enumerate(nb["cells"]):
            if c.get("cell_type") != "code":
                continue
            body = "".join(c.get("source", []))
            if all(f"'{k}':" in body for k in spec["envs"]):
                hits.append(i)
        if len(hits) != 1:
            raise SystemExit(f"expected exactly 1 cell carrying the env anchor, got {hits}")
        en = hits[0]
        body = "".join(nb["cells"][en]["source"])
        for k, v in spec["envs"].items():
            # The anchor is the current 'true' value of the knob; replace exactly once.
            old = f"'{k}': 'true',"
            new = f"'{k}': '{v}',"
            if body.count(old) != 1:
                raise SystemExit(f"expected exactly 1 {old!r}, got {body.count(old)}")
            body = body.replace(old, new)
        nb["cells"][en]["source"] = body.splitlines(keepends=True)
        # The knob is worthless if it does not reach os.environ, and it is unsafe if the
        # cell it landed in also carries the scored-rerun branch.
        if "os.environ.update(" not in body or "setup_env" not in body:
            raise SystemExit("env knob landed in a cell that never exports setup_env")
        # The dict is set before the TRUE_SUBMISSION branch, so both the smoke and the
        # scored rerun read the knob; assert that branch is still in the cell untouched.
        if "TRUE_SUBMISSION = os.environ.get(" not in body:
            raise SystemExit("env knob landed in a cell without the rerun switch")

    # --- the rerun branch must be untouched ----------------------------------------
    patched = "".join(nb["cells"][idx]["source"])
    for required in (
        "if TRUE_SUBMISSION:  # overrides for contest run",
        "bm.solver.concurrency = 120",
        "bm.solver.max_runtime_s_per_game = 532*60",
        "bm.n_passes = 1\n    if USE_PRIORITY_SCHEDULING:",
    ):
        if required not in patched:
            raise SystemExit(f"rerun branch damaged: missing {required!r}")

    # --- every code cell must still parse ------------------------------------------
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        text = "".join(c.get("source", []))
        stripped = text.lstrip()
        # `%%writefile` / `%%bash` cells are not python: their body is a patch or a script.
        if stripped.startswith("%%"):
            continue
        # strip line magics / shell escapes so ast can see the python
        lines = [
            ("" if ln.lstrip().startswith(("%", "!")) else ln)
            for ln in text.splitlines()
        ]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:  # pragma: no cover - build gate
            raise SystemExit(f"cell {i} does not parse: {exc}")

    # --- write the build ------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    ipynb_out = OUT / "submission.ipynb"
    ipynb_out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

    meta = {
        "id": OUR_KERNEL,
        "title": OUR_TITLE,
        "code_file": "submission.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": DATASET_SOURCES,
        "kernel_sources": [],
        "competition_sources": ["arc-prize-2026-arc-agi-3"],
        "model_sources": MODEL_SOURCES,
        "docker_image": DOCKER_IMAGE,
        "machine_shape": "NvidiaRtxPro6000",
    }
    (OUT / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    record = {
        "arm": "duck-sota",
        "variant": variant,
        "what": "verbatim lift of the top-voted public ARC-AGI-3 solution",
        "source_kernel": SOURCE_KERNEL,
        "source_notebook_sha256": src_sha,
        "our_kernel": OUR_KERNEL,
        "kernel_version_pending": True,
        "cells": len(nb["cells"]),
        "patched_cell_index": idx,
        "knobs": spec["cfgs"],
        "diffs": [
            "commit-mode branch (TRUE_SUBMISSION False) bounded to n_passes=1, "
            "concurrency=25, 240 s/game, and the 32400 s budget assertion downgraded "
            "to a warning -- commit-mode smoke only; the scored rerun branch is "
            "untouched and byte-identical to upstream"
        ]
        + ([spec["note"]] if (spec["cfgs"] or spec.get("envs")) else []),
        "machine_shape": "NvidiaRtxPro6000",
        "runtime_cap_min": 540,
        "read_back_key_suggestion": (
            "duck-sota (open-source ARC-AGI-3 lift): Tufa Labs duck harness + "
            "Qwen3.8-Flash-Next; community reads this family at ~27.8 LB"
        ),
    }
    if spec.get("envs"):
        record["smoke_prediction"] = {
            "baseline": {
                "requests": 125, "completion_tokens": 141548,
                "completion_per_request_mean": 1132, "p50": 863, "p90": 2315, "max": 5023,
                "actions": 158, "games_served": "10 of 25",
                "smoke_mean_score": {"v1": 0.46, "batch32": 0.42},
            },
            "lean": ("completion tokens per request fall by more than half (mean below "
                     "~560) while the smoke's mean score holds at or above ~0.30: the "
                     "token bill was reasoning, the action budget roughly doubles, and "
                     "the variant earns the next scored slot"),
            "falsifier": ("mean score below ~0.15 as the tokens fall: reasoning is "
                          "load-bearing, the per-action token axis is closed, and what "
                          "is left for this board is the writeup, not the leaderboard"),
        }

    (OUT / "experiment.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    if "--pull-only" in sys.argv:
        print(pull())
        sys.exit(0)
    build(sys.argv[sys.argv.index("--variant") + 1] if "--variant" in sys.argv else "v1")
