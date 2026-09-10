"""Package the tracked agent with the official offline Kaggle notebook driver."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "out" / "kernel")
    parser.add_argument("--agent", type=Path, default=HERE / "my_agent.py")
    args = parser.parse_args()
    builder = ROOT / "data/ARC-AGI-3-Kaggle-Starter/scripts/build_notebook.py"
    if not builder.is_file():
        parser.error("Official starter missing; run scripts/setup.sh first.")
    spec = importlib.util.spec_from_file_location("arc3_official_builder", builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.AGENT_SRC = args.agent.resolve()
    module.ACCELERATOR = "cpu"
    source = module.AGENT_SRC.read_text(encoding="utf-8")
    compile(source, str(module.AGENT_SRC), "exec")
    notebook = module.build()
    embedded = [cell["source"] for cell in notebook["cells"]
                if cell["cell_type"] == "code"
                and cell["source"].startswith("%%writefile /tmp/my_agent.py\n")]
    if embedded != ["%%writefile /tmp/my_agent.py\n" + source]:
        raise RuntimeError("Official builder did not embed the tracked agent exactly.")
    metadata = json.loads((HERE / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata.update(code_file="submission.ipynb", enable_gpu=False,
                    enable_tpu=False, enable_internet=False)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "submission.ipynb").write_text(
        json.dumps(notebook, indent=1), encoding="utf-8")
    (args.out / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (args.out / "experiment.json").write_text(json.dumps({
        "agent": str(module.AGENT_SRC),
        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "accelerator": "cpu", "internet": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Built {args.out.resolve()} (CPU, offline; tracked agent verified)")


if __name__ == "__main__":
    main()
