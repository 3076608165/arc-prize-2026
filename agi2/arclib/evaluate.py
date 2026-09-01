"""Scoring that mirrors the Kaggle ARC-AGI-2 metric, plus submission I/O.

Kaggle metric: for every *test output* in the evaluation set you supply two
predictions. You score 1 for that output if either prediction matches the
ground truth exactly, else 0. The final score is the mean over all test
outputs (not over tasks -- a task with two test inputs contributes twice).
"""
from __future__ import annotations

import json
from pathlib import Path

from .loader import Grid, Task

# One entry per test input of a task.
Attempts = list[dict[str, Grid]]          # [{"attempt_1": g, "attempt_2": g}, ...]
Submission = dict[str, Attempts]


BLANK: Grid = [[0, 0], [0, 0]]


def make_attempts(preds: list[Grid], n_test: int) -> Attempts:
    """Pad/trim a solver's ranked predictions into Kaggle's 2-attempt shape.

    ``preds`` is the ranked candidate list for a *single* test input.
    """
    a1 = preds[0] if len(preds) > 0 else BLANK
    a2 = preds[1] if len(preds) > 1 else a1
    return {"attempt_1": a1, "attempt_2": a2}


def score(submission: Submission, tasks: list[Task]) -> dict:
    """Return {'score', 'solved_outputs', 'total_outputs', 'per_task': {...}}."""
    total = 0
    hits = 0
    per_task: dict[str, float] = {}
    for task in tasks:
        entry = submission.get(task.task_id, [])
        t_hits = 0
        for i, pair in enumerate(task.test):
            if pair.output is None:
                raise ValueError(f"{task.task_id}: no ground truth to score against")
            total += 1
            got = entry[i] if i < len(entry) else {}
            if pair.output in (got.get("attempt_1"), got.get("attempt_2")):
                hits += 1
                t_hits += 1
        per_task[task.task_id] = t_hits / len(task.test) if task.test else 0.0
    return {
        "score": hits / total if total else 0.0,
        "solved_outputs": hits,
        "total_outputs": total,
        "per_task": per_task,
    }


def validate(submission: Submission, tasks: list[Task]) -> list[str]:
    """Check the structural rules Kaggle enforces. Returns a list of problems."""
    problems: list[str] = []
    for task in tasks:
        if task.task_id not in submission:
            problems.append(f"{task.task_id}: missing from submission")
            continue
        entry = submission[task.task_id]
        if len(entry) != len(task.test):
            problems.append(
                f"{task.task_id}: {len(entry)} entries for {len(task.test)} test inputs"
            )
        for i, att in enumerate(entry):
            for k in ("attempt_1", "attempt_2"):
                g = att.get(k)
                if g is None:
                    problems.append(f"{task.task_id}[{i}]: {k} missing")
                elif not (
                    isinstance(g, list)
                    and g
                    and all(isinstance(r, list) and r for r in g)
                    and len({len(r) for r in g}) == 1
                    and all(isinstance(v, int) and 0 <= v <= 9 for r in g for v in r)
                ):
                    problems.append(f"{task.task_id}[{i}]: {k} is not a valid grid")
    return problems


def write_submission(submission: Submission, path: str | Path = "submission.json") -> Path:
    path = Path(path)
    path.write_text(json.dumps(submission), encoding="utf-8")
    return path
