"""Compare same-policy observation/action trajectories across fresh processes."""
import json
from pathlib import Path
import time

import run
from paired_experiment import environment_fingerprint


def main():
    before = environment_fingerprint()
    started = time.monotonic()
    results = [run.play("stable", ",".join(sorted(run.WIN_LEVELS)), 100, 1000,
                        exact_budget=True, trace=True) for _ in range(2)]
    assert environment_fingerprint() == before, "Offline run changed frozen runtime"
    assert results[0]["traces"] == results[1]["traces"], "Fresh-process trajectories differ"
    assert results[0]["games"] == results[1]["games"], "Fresh-process outcomes differ"
    scores = lambda result: [(r["id"], r["score"]) for r in result["rows"]]
    assert scores(results[0]) == scores(results[1]), "Fresh-process scores differ"
    output = {"status": "passed", "games": 25, "steps_per_game": 100, "policy_seed": 1000,
              "fresh_process_runs": 2, "observation_action_traces_identical": True,
              "runtime_metadata_and_source_unchanged": True,
              "elapsed_seconds": time.monotonic()-started, "traces": results[0]["traces"],
              "limits": "A short determinism regression, not a performance experiment or proof for all later game states."}
    path = Path(__file__).resolve().parents[1] / "experiments/agi3-offline-replay-check.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "traces"}), flush=True)


if __name__ == "__main__":
    main()
