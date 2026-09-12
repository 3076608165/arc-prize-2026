"""Load and fingerprint the competition's public challenges and solutions."""
import hashlib
import json
from pathlib import Path
from arclib.loader import Pair, Task

DEFAULT = Path(__file__).resolve().parents[1] / "data/kaggle-agi2-2026"


def load_public(directory=DEFAULT):
    challenges = directory / "arc-agi_evaluation_challenges.json"
    solutions = directory / "arc-agi_evaluation_solutions.json"
    data = json.loads(challenges.read_text(encoding="utf-8"))
    truth = json.loads(solutions.read_text(encoding="utf-8"))
    if set(data) != set(truth):
        raise ValueError("Challenge and solution task IDs differ")
    tasks = []
    for key, value in sorted(data.items()):
        if len(value["test"]) != len(truth[key]):
            raise ValueError(f"{key}: test input and solution counts differ")
        tasks.append(Task(key, [Pair(p["input"], p["output"]) for p in value["train"]],
                          [Pair(p["input"], output) for p, output in zip(value["test"], truth[key])]))
    provenance = {"source": "Kaggle arc-prize-2026-arc-agi-2 public evaluation files",
                  "tasks": len(tasks), "outputs": sum(len(t.test) for t in tasks),
                  "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (challenges, solutions)}}
    return tasks, provenance
