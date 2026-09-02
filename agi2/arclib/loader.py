"""Load ARC task JSON into plain Python structures.

A task is ``{"train": [{"input": grid, "output": grid}, ...], "test": [...]}``.
A grid is a list of rows of ints 0-9. Test pairs in the *public* data carry an
"output" key; on Kaggle's hidden set they do not.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Grid = list[list[int]]

try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
except NameError:
    # `build_notebook.py` inlines this module into a notebook cell, where
    # `__file__` is undefined. Nothing in the Kaggle driver reads the local
    # dataset paths, so the cwd is a fine stand-in.
    REPO_ROOT = Path.cwd()
AGI2_DATA = REPO_ROOT / "data" / "ARC-AGI-2" / "data"


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid | None


@dataclass(frozen=True)
class Task:
    task_id: str
    train: list[Pair]
    test: list[Pair]


def _pairs(raw: list[dict]) -> list[Pair]:
    return [Pair(input=p["input"], output=p.get("output")) for p in raw]


def load_task(path: str | Path) -> Task:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    return Task(task_id=path.stem, train=_pairs(raw["train"]), test=_pairs(raw["test"]))


def load_split(split: str = "evaluation", root: Path | None = None) -> list[Task]:
    """Load every task in ``training`` or ``evaluation``, sorted by id."""
    root = root or AGI2_DATA
    d = root / split
    if not d.is_dir():
        raise FileNotFoundError(
            f"{d} not found. Run scripts/setup_data.sh to clone the datasets."
        )
    return [load_task(p) for p in sorted(d.glob("*.json"))]


def load_kaggle_challenges(path: str | Path) -> list[Task]:
    """Load Kaggle's single-file format (``*_challenges.json``).

    The hidden test file has no outputs; the public ones ship a matching
    ``*_solutions.json`` keyed the same way.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Task(task_id=tid, train=_pairs(t["train"]), test=_pairs(t["test"]))
        for tid, t in sorted(raw.items())
    ]


def dims(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0]) if g else 0
