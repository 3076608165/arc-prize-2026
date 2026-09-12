"""Submit the user's chosen evidence-gated ARC-AGI-3 policy exactly once."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agi3"))
from paired_experiment import analyze, environment_fingerprint
from submit_when_ready import kaggle

TRIAL = ROOT / "experiments/agi3-coverage-v2-offline"
OUT = ROOT / "experiments/score-push-20260911"


def save(value):
    path = OUT / "agi3-decision.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    decision = OUT / "agi3-decision.json"
    if decision.exists():
        old = json.loads(decision.read_text(encoding="utf-8"))
        if old.get("status") != "waiting_for_complete_experiment":
            raise SystemExit("Prior upload/submission intent exists; inspect server before any restart.")
    save({"status": "waiting_for_complete_experiment",
          "authorization": "User: finish experiment, submit candidate if predeclared criteria pass; otherwise conservative baseline."})
    for _ in range(960):
        protocol = json.loads((TRIAL / "protocol.json").read_text(encoding="utf-8"))
        results_path = TRIAL / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            assessment = analyze(protocol, results)
            if assessment["status"] != "collecting":
                break
        time.sleep(45)
    else:
        raise SystemExit("Experiment still incomplete after 12 hours; no submission attempted.")
    if environment_fingerprint() != protocol["environment"]:
        raise ValueError("Trial runtime fingerprint changed; no automatic decision")
    arm = "coverage" if assessment["status"] == "eligible_for_online_experiment" else "stable"
    bundle = ROOT / f"agi3/out/{arm}-20260911"
    notebook_path = bundle / "submission.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    snapshot_path = TRIAL / protocol["arms"][arm]["source"]
    snapshot = snapshot_path.read_text(encoding="utf-8")
    if hashlib.sha256(snapshot_path.read_bytes()).hexdigest() != protocol["arms"][arm]["sha256"]:
        raise ValueError("Frozen selected source changed")
    sources = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    if sources.count("%%writefile /tmp/my_agent.py\n" + snapshot) != 1:
        raise ValueError("Selected notebook does not embed the assessed source")
    metadata = json.loads((bundle / "kernel-metadata.json").read_text(encoding="utf-8"))
    if metadata["id"] != "uts3125025522741/arc-agi-3-sweep-agent" or not metadata["is_private"]:
        raise ValueError("Unexpected destination")
    record = {"status": "pushing", "arm": arm, "assessment": assessment,
              "source_sha256": protocol["arms"][arm]["sha256"],
              "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest()}
    save(record)
    response = kaggle("kernels", "push", "-p", str(bundle))
    match = re.search(r"Kernel version (\d+) successfully pushed", response)
    if not match:
        record.update(status="push_uncertain", response=response)
        save(record)
        raise SystemExit("Cannot determine accepted version; inspect Kaggle, do not retry push.")
    version = int(match.group(1))
    record.update(status="uploaded_waiting_for_commit", kernel=metadata["id"], version=version,
                  response=response)
    save(record)
    message = f"20260911 {arm} v{version}: complete 12-pair offline gate; " + (
        "eligible online experiment, no hidden improvement claim" if arm == "coverage"
        else "conservative fallback; candidate did not meet predeclared gate")
    subprocess.run([sys.executable, str(ROOT / "scripts/submit_when_ready.py"),
        "--kernel", metadata["id"], "--version", str(version),
        "--competition", "arc-prize-2026-arc-agi-3", "--file", "submission.parquet",
        "--message", message, "--receipt", str(OUT / "agi3-receipt.json"), "--timeout", "10800"],
        cwd=ROOT, check=True)
    record["status"] = "submission_confirmed"
    save(record)
    print(json.dumps({"status": record["status"], "arm": arm, "version": version}), flush=True)


if __name__ == "__main__":
    main()
