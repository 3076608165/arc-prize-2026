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


def build() -> dict:
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
        "what": "verbatim lift of the top-voted public ARC-AGI-3 solution",
        "source_kernel": SOURCE_KERNEL,
        "source_notebook_sha256": src_sha,
        "our_kernel": OUR_KERNEL,
        "kernel_version_pending": True,
        "cells": len(nb["cells"]),
        "patched_cell_index": idx,
        "diffs": [
            "commit-mode branch (TRUE_SUBMISSION False) bounded to n_passes=1, "
            "concurrency=25, 240 s/game, and the 32400 s budget assertion downgraded "
            "to a warning -- commit-mode smoke only; the scored rerun branch is "
            "untouched and byte-identical to upstream"
        ],
        "machine_shape": "NvidiaRtxPro6000",
        "runtime_cap_min": 540,
        "read_back_key_suggestion": (
            "20261004 duck-sota (open-source lift): top-voted public ARC-AGI-3 "
            "solution (Tufa Labs duck harness + Qwen3.8-Flash-Next), verbatim except a "
            "bounded commit-mode smoke; community reads this family at ~27.8 LB"
        ),
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
    build()
