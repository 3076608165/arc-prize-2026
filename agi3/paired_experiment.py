"""Freeze, run, and assess one predeclared paired public-game experiment."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import run

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "experiments" / "agi3-coverage-v2-offline"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def environment_fingerprint():
    starter = run.STARTER
    site = starter / ".venv/Lib/site-packages"
    if not site.exists():
        candidates = sorted((starter / ".venv/lib").glob("python*/site-packages"))
        if not candidates:
            raise ValueError("Cannot find the installed game runtime")
        site = candidates[0]
    files = [starter / "scripts/play_local.py", starter / ".venv/pyvenv.cfg", Path(run.__file__),
             Path(__file__)]
    for directory in [starter / "environment_files", starter / "vendor/ARC-AGI-3-Agents/agents",
                      site / "arc_agi", site / "arcengine"]:
        if not directory.exists():
            raise ValueError(f"Missing runtime source directory: {directory}")
        files.extend(p for p in directory.rglob("*") if p.suffix in (".py", ".json"))
    for package in ("arc_agi", "arcengine", "numpy", "pydantic"):
        files.extend(site.glob(f"{package}-*.dist-info/METADATA"))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(files))}


def create(directory):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "protocol.json").exists():
        raise ValueError("Protocol already exists; do not change a running experiment")
    sources = directory / "sources"
    sources.mkdir(exist_ok=True)
    arms = {}
    for arm in ("stable", "coverage"):
        target = sources / f"{arm}.py"
        target.write_text(run.AGENTS[arm].read_text(encoding="utf-8"), encoding="utf-8")
        arms[arm] = {"source": f"sources/{arm}.py", "sha256": digest(target)}
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Resampling visible objects for at most one in ten clicks, while preserving a full pixel sweep, improves mean public-game score over the same-seeded sweep control.",
        "arms": arms, "control": "stable", "candidate": "coverage",
        "seeds": list(range(1000, 1012)), "games": sorted(run.WIN_LEVELS),
        "action_budget": 5000, "exact_budget": True,
        "environment_mode": "OFFLINE; sorted game order; no online environment refresh",
        "environment_randomness": "Engine seed 0; Python/NumPy global streams reset before each game using SHA256(ARC_ENV_V1:game_id) first four bytes. Policy seeds alone vary across pairs.",
        "supersedes": "agi3-coverage-v1 failed before saving any result because NORMAL mode refreshed metadata. Same agents, seeds, budget and decision thresholds; no performance result was inspected.",
        "primary_metric": "Mean engine score over all 25 games, then mean over 12 paired policy seeds.",
        "order": "Alternate control/candidate order by seed index; no adaptive stopping or seed extension.",
        "minimum_relative_gain": 0.15, "minimum_absolute_gain": 0.02,
        "minimum_positive_pairs": 8, "alpha": 0.05,
        "bootstrap_draws": 20000, "analysis_seed": 20260910,
        "advance_rule": "All 12 valid pairs; paired 95% bootstrap lower bound > 0; two-sided sign-flip p <= .05; mean gain >= max(15% of control, 0.02 points); at least 8 positive pairs. Advance means eligible for an online experiment, not demonstrated hidden-set improvement.",
        "inconclusive_rule": "If the advance rule is not met, do not promote or enlarge the seed set after inspecting results. Missing jobs or changed sources/runtime invalidate the round.",
        "limitations": ["Twelve seeds is a fixed compute budget, not a claim of sufficient power.",
                        "Bootstrap uncertainty describes policy randomness conditional on these fixed public games; it does not cover hidden-game generalization.",
                        "The paired sign-flip test assumes exchangeability/symmetry under its null.",
                        "Local game mode and the Kaggle gateway are different execution settings.",
                        "These are new seed-matched artifacts, not a rerun of the previous objects-v3 experiment."],
        "environment": environment_fingerprint(),
    }
    write_json(directory / "protocol.json", protocol)
    print(f"Frozen {len(protocol['seeds'])} pairs, {len(protocol['games'])} games, {protocol['action_budget']} actions/game: {directory}")


def percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def paired_statistics(control, candidate, draws=20000, seed=20260910):
    if len(control) != len(candidate) or not control:
        raise ValueError("Need complete pairs")
    differences = [b-a for a, b in zip(control, candidate)]
    n = len(differences)
    observed = statistics.mean(differences)
    extremes = sum(abs(sum(s*d for s, d in zip(signs, differences))/n) >= abs(observed)-1e-12
                   for signs in itertools.product((-1, 1), repeat=n))
    rng = random.Random(seed)
    bootstrap = [statistics.mean(rng.choices(differences, k=n)) for _ in range(draws)]
    return {"control_mean": statistics.mean(control), "candidate_mean": statistics.mean(candidate),
            "mean_paired_difference": observed, "paired_differences": differences,
            "positive_pairs": sum(d > 0 for d in differences),
            "sign_flip_p_two_sided": extremes / 2**n,
            "paired_bootstrap_ci95": [percentile(bootstrap, .025), percentile(bootstrap, .975)]}


def analyze(protocol, records):
    expected = {(seed, arm) for seed in protocol["seeds"] for arm in protocol["arms"]}
    found = {(r["seed"], r["arm"]) for r in records}
    if len(found) != len(records) or not found <= expected:
        raise ValueError("Duplicate or unexpected runs")
    rows_by_key = {}
    game_versions = None
    for record in records:
        arm = record["arm"]
        if record["source_sha256"] != protocol["arms"][arm]["sha256"]:
            raise ValueError("Source hash mismatch")
        result = record["result"]
        run.validate_results(result["games"], result["rows"], protocol["games"])
        if result.get("exact_budget") is not True:
            raise ValueError("Undeclared budget convention")
        if any(g["actions"] > protocol["action_budget"] for g in result["games"].values()):
            raise ValueError("Action budget exceeded")
        versions = sorted(row.get("id") or row["game_id"] for row in result["rows"])
        if game_versions is not None and versions != game_versions:
            raise ValueError("Game versions differ across runs")
        game_versions = versions
        rows_by_key[record["seed"], arm] = run.true_score(result["rows"])
    if found != expected:
        return {"status": "collecting", "completed_runs": len(found), "required_runs": len(expected)}
    control = [rows_by_key[s, protocol["control"]] for s in protocol["seeds"]]
    candidate = [rows_by_key[s, protocol["candidate"]] for s in protocol["seeds"]]
    stats = paired_statistics(control, candidate, protocol["bootstrap_draws"], protocol["analysis_seed"])
    threshold = max(protocol["minimum_absolute_gain"], stats["control_mean"] * protocol["minimum_relative_gain"])
    advance = (stats["mean_paired_difference"] >= threshold
               and stats["paired_bootstrap_ci95"][0] > 0
               and stats["sign_flip_p_two_sided"] <= protocol["alpha"]
               and stats["positive_pairs"] >= protocol["minimum_positive_pairs"])
    return {"status": "eligible_for_online_experiment" if advance else "inconclusive_do_not_promote",
            "required_mean_gain": threshold, "game_versions": game_versions,
            "control_scores": control, "candidate_scores": candidate, **stats,
            "limitations": protocol["limitations"]}


def execute(directory):
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    results_path = directory / "results.json"
    records = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    analyze(protocol, records)
    completed = {(r["seed"], r["arm"]) for r in records}
    for index, seed in enumerate(protocol["seeds"]):
        arms = [protocol["control"], protocol["candidate"]]
        if index % 2:
            arms.reverse()
        for arm in arms:
            if (seed, arm) in completed:
                continue
            if environment_fingerprint() != protocol["environment"]:
                raise ValueError("Runtime changed; this round cannot be continued")
            spec = protocol["arms"][arm]
            path = directory / spec["source"]
            if digest(path) != spec["sha256"]:
                raise ValueError("Frozen agent source changed")
            run.AGENTS["paired_" + arm] = path
            result = run.play("paired_" + arm, ",".join(protocol["games"]),
                              protocol["action_budget"], seed, exact_budget=True)
            run.validate_results(result["games"], result["rows"], protocol["games"])
            if digest(path) != spec["sha256"] or environment_fingerprint() != protocol["environment"]:
                raise ValueError("Source/runtime changed during execution")
            records.append({"arm": arm, "seed": seed, "source_sha256": spec["sha256"], "result": result})
            write_json(results_path, records)
            assessment = analyze(protocol, records)
            write_json(directory / "assessment.json", assessment)
            print(f"Completed {len(records)}/{len(protocol['seeds'])*2} runs; status={assessment['status']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "run", "analyze"))
    parser.add_argument("--directory", type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == "create":
        create(args.directory)
    elif args.action == "run":
        execute(args.directory)
    else:
        p = json.loads((args.directory / "protocol.json").read_text(encoding="utf-8"))
        r = json.loads((args.directory / "results.json").read_text(encoding="utf-8"))
        assessment = analyze(p, r)
        write_json(args.directory / "assessment.json", assessment)
        print(json.dumps(assessment, indent=2))
