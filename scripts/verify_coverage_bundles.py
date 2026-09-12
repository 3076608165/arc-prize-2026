"""Verify prepared artifacts without executing notebooks or starting experiments."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agi3"))
from paired_experiment import environment_fingerprint


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    protocol_directory = ROOT / "experiments/agi3-coverage-v2-offline"
    protocol = json.loads((protocol_directory / "protocol.json").read_text(encoding="utf-8"))
    assert environment_fingerprint() == protocol["environment"], "Frozen AGI-3 runtime changed"
    bundles = []
    for track, folders in (("agi2", ["audited16", "audited8", "coverage-benchmark16", "coverage-benchmark8"]),
                           ("agi3", ["stable-control", "coverage-candidate"])):
        for folder in folders:
            directory = ROOT / track / "out" / folder
            notebook_path = directory / "submission.ipynb"
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            experiment = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
            modules, compiled, skipped = {}, [], []
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                if source.startswith("%%writefile "):
                    header, source = source.split("\n", 1)
                    modules[header.split()[-1]] = source
                elif any(line.lstrip().startswith(("!", "%")) for line in source.splitlines()):
                    skipped.append(index)
                    continue
                compile(source, f"{folder}:cell{index}", "exec")
                compiled.append(index)
            metadata = json.loads((directory / "kernel-metadata.json").read_text(encoding="utf-8"))
            assert metadata["enable_internet"] is False
            if track == "agi2":
                assert digest(notebook_path) == experiment["notebook_sha256"]
                assert modules["coverage_audit.py"] == (ROOT / "agi2/coverage_audit.py").read_text(encoding="utf-8")
                assert f"train_ds = puzzle_ds.augment(n={experiment['train_augment']}" in modules["arc_solver.py"]
                assert experiment["worker_hash_seed"] == 0
                assert metadata["machine_shape"] == "NvidiaL4"
                assert metadata.get("docker_image") and metadata.get("model_sources")
                if folder.startswith("coverage-benchmark"):
                    assert "if key not in" not in modules["starter.py"]
                    assert experiment["hours"] == 1 and experiment["commit_subset"] == "all"
            else:
                arm = "stable" if folder == "stable-control" else "coverage"
                agent = (ROOT / "agi3" / f"{arm}_agent.py").read_text(encoding="utf-8")
                snapshot = protocol_directory / protocol["arms"][arm]["source"]
                assert modules["/tmp/my_agent.py"] == agent == snapshot.read_text(encoding="utf-8")
                assert hashlib.sha256(agent.encode()).hexdigest() == experiment["sha256"]
                assert digest(snapshot) == protocol["arms"][arm]["sha256"]
            bundles.append({"directory": directory.relative_to(ROOT).as_posix(),
                            "notebook_sha256": digest(notebook_path),
                            "metadata_sha256": digest(directory / "kernel-metadata.json"),
                            "compiled_python_cells": compiled, "unvalidated_shell_magic_cells": skipped,
                            "execution_status": "prepared_only"})
    return {"bundles": bundles, "agi3_frozen_runtime_matches": True,
            "limits": "Embedded Python syntax and source identity checked. Shell/magic execution and full Kaggle runtime are not validated by this check."}


if __name__ == "__main__":
    report = verify()
    destination = ROOT / "experiments/coverage-delivery-verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(report['bundles'])} bundles; AGI-3 frozen runtime unchanged. {destination}")
