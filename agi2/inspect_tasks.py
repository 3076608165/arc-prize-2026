"""Render tasks (and optionally the baseline's guesses) to a browsable HTML page.

    python agi2/inspect_tasks.py --split evaluation --limit 12
    python agi2/inspect_tasks.py --task 0934a4d8 --predict
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arclib.loader import load_split
from arclib.solvers import solve_task
from arclib.viz import task_to_html, write_html


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    ap.add_argument("--task", default="", help="a single task id")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--predict", action="store_true", help="overlay baseline guesses")
    ap.add_argument("--out", default="agi2/out/tasks.html")
    ap.add_argument("--cell", type=int, default=14)
    args = ap.parse_args()

    tasks = load_split(args.split)
    if args.task:
        tasks = [t for t in tasks if t.task_id == args.task]
        if not tasks:
            print(f"no task {args.task} in {args.split}")
            return 1
    else:
        tasks = tasks[: args.limit]

    body = []
    for t in tasks:
        preds = None
        if args.predict:
            preds, names = solve_task(t.train, [p.input for p in t.test])
            title = f"{t.task_id}  [{', '.join(names) or 'no rule fits'}]"
        else:
            title = t.task_id
        body.append(task_to_html(t, preds, cell=args.cell, title=title))

    p = write_html("<hr style='border-color:#333;margin:32px 0'>".join(body),
                   args.out, title=f"ARC-AGI-2 {args.split}")
    print(f"wrote {p.resolve()}  ({len(tasks)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
