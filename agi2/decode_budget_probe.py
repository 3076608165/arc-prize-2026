"""Replay one frozen, label-free decode stopping rule against cached candidates.

This does not change the upstream selector or any running notebook. The policy
receives only the first eight (non-transposed) decoded views. Ground truth is
loaded by the evaluator only after every output's stopping decision is fixed.
"""
from __future__ import annotations

import argparse
import bz2
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle

from arclib.loader import load_split
from competition_data import load_public

ROOT = Path(__file__).resolve().parents[1]
POLICY = {
    "name": "unanimous-first-eight-v1",
    "stage": "Non-transposed views: two permutations of each of four rotations",
    "minimum_nonempty_views": 8,
    "per_rotation_views": 2,
    "gate": [
        "Exactly eight first-stage views exist and each contains valid candidates.",
        "The minimum-beam-score grid agrees exactly across all eight views.",
        "The upstream top-two grid set agrees between rotations {0,2} and {1,3}.",
    ],
    "action": "Keep upstream top-two selection on these eight views and omit the two later transposed-view batches for this output.",
    "selector": "Frozen upstream score_full_probmul_3; unchanged ranking formula",
    "threshold_search": False,
    "test_truth_used_by_policy": False,
}


def grid_key(grid):
    return tuple(tuple(int(cell) for cell in row) for row in grid)


def geometry(name):
    operations = name.split(".")[1:]
    return "transpose" in operations, operations.count("rot90")


def flatten(views):
    return {f"{name}.out{index}": sample
            for name, samples in sorted(views.items())
            for index, sample in enumerate(samples)}


def decide(first_stage_views, selector):
    """The policy API has no full pool, test task, output label, or task ID."""
    if any(geometry(name)[0] for name in first_stage_views):
        raise ValueError("Future transposed views must not enter the stopping policy")
    counts = Counter(geometry(name)[1] for name in first_stage_views)
    if counts != Counter({rotation: 2 for rotation in range(4)}):
        return {"stop": False, "reason": "incomplete_first_stage"}
    if any(not samples for samples in first_stage_views.values()):
        return {"stop": False, "reason": "empty_first_stage_view"}
    view_winners = {grid_key(min(samples, key=lambda sample: float(sample["beam_score"]))["solution"])
                    for samples in first_stage_views.values()}
    if len(view_winners) != 1:
        return {"stop": False, "reason": "view_winner_disagreement"}
    halves = [{name: samples for name, samples in first_stage_views.items()
               if geometry(name)[1] % 2 == parity} for parity in (0, 1)]
    choices = [{grid_key(grid) for grid in selector(flatten(half))[:2]} for half in halves]
    if choices[0] != choices[1]:
        return {"stop": False, "reason": "batch_top_two_disagreement"}
    return {"stop": True, "reason": "unanimous_first_eight_and_stable_top_two"}


def load_store(store):
    views = defaultdict(dict)
    file_hashes = {}
    for path in sorted(store.iterdir()):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        file_hashes[path.name] = hashlib.sha256(raw).hexdigest()
        samples = pickle.loads(bz2.decompress(raw))
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Invalid saved candidate file: {path.name}")
        for sample in samples:
            if not {"solution", "beam_score", "score_aug"} <= sample.keys():
                raise ValueError(f"Incomplete candidate in {path.name}")
        views[path.name.split(".")[0]][path.name] = samples
    fingerprint = hashlib.sha256(json.dumps(file_hashes, sort_keys=True).encode()).hexdigest()
    return dict(views), {"files": len(file_hashes), "manifest_sha256": fingerprint}


def load_selector(path):
    spec = importlib.util.spec_from_file_location("frozen_decode_budget_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.score_full_probmul_3


def freeze(path, store, decoder):
    if path.exists():
        raise ValueError("Protocol exists; do not silently replace a predeclared rule")
    _, fingerprint = load_store(store)
    protocol = {
        "evaluation_data": load_public()[1],
        "revision_note": "v2 corrects the evaluation dataset to Kaggle public files. The v1 Git-data evaluation failed before any result; the stopping rule and thresholds are unchanged.",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": POLICY,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "store": str(store.resolve()),
        "store_fingerprint": fingerprint,
        "decoder": str(decoder.resolve()),
        "decoder_sha256": hashlib.sha256(decoder.read_bytes()).hexdigest(),
        "outcomes": ["number of stopping decisions", "known later emitting views omitted",
                     "top-two exact wins/losses against unchanged full cached pool"],
        "diagnostic_gate": "At least 10 eligible outputs, zero lost exact outputs, and at least 10% of all known emitting views omitted before proposing a prospective GPU test; no automatic promotion.",
        "limitations": [
            "Single historical public pool, already inspected by prior experiments; not an untouched holdout.",
            "Missing candidate files may mean no valid completion or a truncated/unstarted batch; absence is not measured runtime.",
            "Counted saved emitting views cannot be converted to GPU seconds without an audited prospective run.",
            "Replaying unchanged candidates cannot simulate extra tail tasks or altered deadline allocation.",
            "Agreement can repeat a wrong hypothesis; it is a stopping heuristic, not a correctness certificate.",
            "Stopping removes later scores that might rank a correct answer differently; per-output losses must be reported.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    return protocol


def evaluate(protocol_path, output_path):
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["policy"] != POLICY or protocol["implementation_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("Policy or implementation changed after protocol freeze")
    decoder = Path(protocol["decoder"])
    if protocol["decoder_sha256"] != hashlib.sha256(decoder.read_bytes()).hexdigest():
        raise ValueError("Upstream decoder changed after protocol freeze")
    stored, fingerprint = load_store(Path(protocol["store"]))
    if fingerprint != protocol["store_fingerprint"]:
        raise ValueError("Candidate pool changed after protocol freeze")
    selector = load_selector(decoder)
    # Freeze every policy decision before the evaluation labels are loaded.
    decisions = {}
    predictions = {}
    for base, views in sorted(stored.items()):
        first = {name: samples for name, samples in views.items() if not geometry(name)[0]}
        decision = decide(first, selector)
        later = {name: samples for name, samples in views.items() if geometry(name)[0]}
        decision.update(first_emitting_views=len(first), later_emitting_views=len(later),
                        observed_emitting_views_omitted=len(later) if decision["stop"] else 0)
        decisions[base] = decision
        full = [grid_key(grid) for grid in selector(flatten(views))[:2]]
        stopped = [grid_key(grid) for grid in selector(flatten(first))[:2]] if decision["stop"] else full
        predictions[base] = {"baseline": full, "candidate": stopped}
    decision_sha = hashlib.sha256(json.dumps(decisions, sort_keys=True).encode()).hexdigest()
    tasks, provenance = load_public()
    if provenance != protocol["evaluation_data"]:
        raise ValueError("Competition evaluation data changed")
    truth = {f"{task.task_id}_{index}": grid_key(pair.output)
             for task in tasks for index, pair in enumerate(task.test)}
    if not set(stored) <= set(truth):
        raise ValueError("Pool has outputs outside the declared evaluation split")
    correct = {arm: {base for base, pred in predictions.items() if truth[base] in pred[arm]}
               for arm in ("baseline", "candidate")}
    stopped = [base for base, decision in decisions.items() if decision["stop"]]
    gained = sorted(correct["candidate"] - correct["baseline"])
    lost = sorted(correct["baseline"] - correct["candidate"])
    omitted = sum(row["observed_emitting_views_omitted"] for row in decisions.values())
    fraction = omitted / fingerprint["files"] if fingerprint["files"] else 0
    per_task = {arm: sum(sum(f"{task.task_id}_{i}" in keys for i in range(len(task.test))) / len(task.test)
                         for task in tasks) / len(tasks) for arm, keys in correct.items()}
    report = {
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "policy_decisions_sha256_before_loading_truth": decision_sha,
        "candidate_pool": fingerprint,
        "outputs_in_pool": len(stored), "all_public_outputs": len(truth),
        "stopped_outputs": len(stopped), "stopped_output_ids": stopped,
        "reason_counts": dict(Counter(row["reason"] for row in decisions.values())),
        "observed_emitting_views_omitted": omitted,
        "observed_emitting_view_reduction": fraction,
        "planned_later_views_at_most_omitted": 8 * len(stopped),
        "exact_outputs": {arm: len(keys) for arm, keys in correct.items()},
        "exact_task_weighted_fraction_over_all_120": per_task,
        "gained_outputs": gained, "lost_outputs": lost,
        "unprocessed_public_outputs_counted_as_failure": len(truth) - len(stored),
        "eligible_for_prospective_timing_test": len(stopped) >= 10 and not lost and fraction >= .10,
        "established_score_improvement": False,
        "limitations": protocol["limitations"],
        "decisions": decisions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/agi2-decode-budget-v2/protocol.json")
    parser.add_argument("--store", type=Path, default=ROOT / "agi2/out/v7/inference_outputs")
    parser.add_argument("--decoder", type=Path, default=ROOT / "agi2/out/v7/arc_decoder.py")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/agi2-decode-budget-v2/result.json")
    args = parser.parse_args()
    result = freeze(args.protocol, args.store, args.decoder) if args.mode == "freeze" else evaluate(args.protocol, args.out)
    print(json.dumps({key: value for key, value in result.items() if key not in {"decisions", "limitations", "policy"}}, indent=2))


if __name__ == "__main__":
    main()
