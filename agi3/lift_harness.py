"""Lift the public ARC-AGI-3 gemma-reflection harness into our kernel as an arm.

Source: `bernicerobinson/arc-agi-3-m2-gemma-reflection` (pulled to
`experiments/intel-20260926/arc-agi-3-m2-gemma-reflection/`) — offline
(`enable_internet=False`), GPU, attaches `google/gemma-4/.../gemma-4-31b-it/1`,
`ko0kip/vllm-0230-offline` and the `philipvonderlind/vllm-deps` kernel, and drives
the official ARC-AGI-3-Agents harness with its own `MyAgent`.

Lifted, not rewritten: the agent IS the arm (see
`references/kernel-experiment-runs.md`, "Lifting a public arm's cells"). The tracked
artifact is `agi3/harness_gemma_my_agent.py`; the build directory is generated from
it, so a knob is an edit to our file and the provenance hash is re-checked on every
build instead of trusting the pulled notebook.

    python agi3/lift_harness.py                # extract (first run) + build
    python agi3/lift_harness.py --self-test    # checks only, writes nothing

The build cannot be pushed until the agent file exists, and it refuses if the pulled
notebook's agent cell no longer matches our tracked copy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE_REF = "bernicerobinson/arc-agi-3-m2-gemma-reflection"
SOURCE_DIR = ROOT / "experiments" / "intel-20260926" / "arc-agi-3-m2-gemma-reflection"
AGENT_FILE = HERE / "harness_gemma_my_agent.py"
OUR_ID = "uts3125025522741/arc-agi-3-sweep-agent"
AGENT_HEADER = "%%writefile /kaggle/working/my_agent.py\n"


def source_notebook() -> pathlib.Path:
    nbs = sorted(SOURCE_DIR.glob("*.ipynb"))
    if not nbs:
        raise SystemExit(f"pulled source missing: {SOURCE_DIR} (kaggle kernels pull {SOURCE_REF})")
    return nbs[0]


def agent_cell(nb: dict) -> tuple[int, str]:
    """The (index, cell source) of the harness's agent cell, or raise."""
    hits = [(i, "".join(c["source"])) for i, c in enumerate(nb["cells"])
            if c["cell_type"] == "code" and "".join(c["source"]).startswith(AGENT_HEADER)]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one agent cell starting {AGENT_HEADER!r}, found {len(hits)}")
    return hits[0]


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=HERE / "out" / "harness-gemma")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    src_nb_path = source_notebook()
    nb = json.loads(src_nb_path.read_text(encoding="utf-8"))
    idx, cell_source = agent_cell(nb)
    pulled_agent = cell_source[len(AGENT_HEADER):]
    if not pulled_agent.strip():
        raise SystemExit("agent cell is empty")

    if args.self_test:
        assert AGENT_HEADER.endswith("\n")
        assert OUR_ID.endswith("arc-agi-3-sweep-agent")
        print(f"self-test ok: source {src_nb_path.name}, agent cell #{idx}, "
              f"{len(pulled_agent)} chars, {len([c for c in nb['cells'] if c['cell_type'] == 'code'])} code cells")
        return 0

    if not AGENT_FILE.exists():
        AGENT_FILE.write_text(pulled_agent, encoding="utf-8", newline="")
        print(f"extracted the lifted agent -> {AGENT_FILE.name} ({len(pulled_agent)} chars)")
    elif AGENT_FILE.read_bytes() != pulled_agent.encode("utf-8"):
        raise SystemExit(f"{AGENT_FILE.name} differs from the pulled agent cell: our copy is edited, "
                         "so either re-pull and re-review or record the divergence deliberately")

    meta_src = SOURCE_DIR / "kernel-metadata.json"
    meta = json.loads(meta_src.read_text(encoding="utf-8"))
    meta.update(id=OUR_ID, title="ARC AGI 3 sweep agent", code_file="submission.ipynb",
                is_private=True)
    meta.pop("id_no", None)

    # The build's agent cell is OUR tracked file, byte for byte.
    nb["cells"][idx]["source"] = (AGENT_HEADER + AGENT_FILE.read_text(encoding="utf-8")).splitlines(keepends=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "submission.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (args.out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    built = json.loads((args.out / "submission.ipynb").read_text(encoding="utf-8"))
    agent_cells = ["".join(c["source"]) for c in built["cells"]
                   if c["cell_type"] == "code" and "".join(c["source"]).startswith(AGENT_HEADER)]
    assert agent_cells == [AGENT_HEADER + AGENT_FILE.read_text(encoding="utf-8")], \
        "built notebook does not embed the tracked agent exactly"

    record = {
        "arm": "gemma-reflection harness (lifted)",
        "source_ref": SOURCE_REF,
        "source_notebook_sha256": sha256(src_nb_path),
        "lifted_agent_sha256": sha256(AGENT_FILE),
        "built_notebook_sha256": sha256(args.out / "submission.ipynb"),
        "agent_cell_index": idx,
        "attach": {"model_sources": meta.get("model_sources"),
                   "dataset_sources": meta.get("dataset_sources"),
                   "kernel_sources": meta.get("kernel_sources")},
        "machine_shape": meta.get("machine_shape"),
        "docker_image": meta.get("docker_image"),
        "note": "third-party arm, lifted verbatim: our only change is the kernel identity. "
                "The commit-mode run exercises the environment (vllm + arc_agi + the official "
                "agents harness) and writes a dummy submission.parquet; the agent loop itself "
                "only runs in the competition rerun, so the first scored slot is the first "
                "measurement of the agent.",
    }
    (args.out / "experiment.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print(f"\nwrote {args.out}\npush with:\n  kaggle kernels push -p {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
