"""Compare downloaded public deadline runs, separating coverage from accuracy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arclib.evaluate import score, validate
from arclib.loader import load_split
from coverage_audit import summarize
from deadline_coverage import summarize_deadline
from competition_data import load_public, DEFAULT


def load_run(directory, tasks):
    report_path = directory / "coverage_summary.json"
    audit = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else summarize(directory / "coverage_audit")
    if audit.get("empty_audit") or audit.get("event_errors"):
        raise ValueError(f"{directory}: absent or invalid coverage audit")
    if set(audit["tasks"]) != {t.task_id for t in tasks}:
        raise ValueError("Audit queue must match the entire public evaluation set")
    submission = json.loads((directory / "submission.json").read_text(encoding="utf-8"))
    problems = validate(submission, tasks)
    if problems:
        raise ValueError(f"Invalid submission: {problems[:5]}")
    hits = {}
    for task in tasks:
        for index, pair in enumerate(task.test):
            attempts = submission[task.task_id][index]
            hits[f"{task.task_id}_{index}"] = pair.output in (attempts["attempt_1"], attempts["attempt_2"])
    covered = {out for task in audit["tasks"].values() for out in task["candidate_output_ids"]}
    if not covered <= set(hits):
        raise ValueError("Audit contains output IDs outside the public data")
    return audit, score(submission, tasks), hits, covered


def compare(baseline, candidate, tasks):
    a, am, ah, ac = load_run(baseline, tasks)
    b, bm, bh, bc = load_run(candidate, tasks)
    common = sorted(ac & bc)
    gained = [key for key in common if bh[key] and not ah[key]]
    lost = [key for key in common if ah[key] and not bh[key]]
    ad = summarize_deadline(baseline / "coverage_audit", ac)
    bd = summarize_deadline(candidate / "coverage_audit", bc)
    deadline_complete = ad["status"] == bd["status"] == "complete"
    return {
        "baseline": {"coverage": a["counts"], "deadline_coverage": ad, "all_public_outputs": am},
        "candidate": {"coverage": b["counts"], "deadline_coverage": bd, "all_public_outputs": bm},
        "coverage_difference_outputs": len(bc) - len(ac),
        "deadline_coverage_difference_outputs": (
            bd["counts"]["confirmed_by_deadline"] - ad["counts"]["confirmed_by_deadline"]
            if deadline_complete else None),
        "common_candidate_outputs": {"n": len(common), "baseline_correct": sum(ah[k] for k in common),
                                     "candidate_correct": sum(bh[k] for k in common), "gained": gained, "lost": lost},
        "candidate_only_covered": sorted(bc-ac), "baseline_only_covered": sorted(ac-bc),
        "interpretation": "Descriptive diagnostic for the fixed public queue. Overall candidate coverage includes late writes; deadline coverage is separately reconstructed from original logs, and its difference is unknown if either audit is incomplete. Final submission accuracy may use late candidates, so it is not accuracy at the deadline. Candidate coverage does not mean correct answers. The common processed subset is selected by runtime, and test outputs within a task are correlated. One run per arm does not establish superiority or hidden-set coverage."
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT)
    args = parser.parse_args()
    tasks, provenance = load_public(args.data)
    report = compare(args.baseline, args.candidate, tasks)
    report["evaluation_data"] = provenance
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"coverage_difference_outputs": report["coverage_difference_outputs"],
                      "common_candidate_outputs": report["common_candidate_outputs"]}, indent=2))
