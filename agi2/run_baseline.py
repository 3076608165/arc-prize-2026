"""Run the baseline portfolio over a split and report the Kaggle metric.

    python agi2/run_baseline.py --split evaluation
    python agi2/run_baseline.py --split training --limit 200
    python agi2/run_baseline.py --split evaluation --out agi2/out/submission.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from arclib.evaluate import make_attempts, score, validate, write_submission
from arclib.loader import load_split
from arclib.solvers import rule_names, solve_task, solve_task_full


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["evaluation", "training"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N tasks")
    ap.add_argument("--out", default="", help="write submission.json here")
    ap.add_argument("--verbose", action="store_true", help="list every solved task")
    ap.add_argument("--full", action="store_true",
                    help="rule portfolio + compositional search (slower)")
    ap.add_argument("--depth", type=int, default=3, help="search depth for --full")
    args = ap.parse_args()

    tasks = load_split(args.split)
    if args.limit:
        tasks = tasks[: args.limit]

    t0 = time.perf_counter()
    submission = {}
    fired: dict[str, int] = {n: 0 for n in rule_names()}
    from collections import defaultdict
    fired = defaultdict(int, fired)
    for task in tasks:
        if args.full:
            per_test, names = solve_task_full(
                task.train, [p.input for p in task.test], max_depth=args.depth
            )
        else:
            per_test, names = solve_task(task.train, [p.input for p in task.test])
        for n in names:
            fired[n] += 1
        submission[task.task_id] = [
            make_attempts(preds, len(task.test)) for preds in per_test
        ]
    elapsed = time.perf_counter() - t0

    problems = validate(submission, tasks)
    result = score(submission, tasks)

    print(f"split           : {args.split} ({len(tasks)} tasks)")
    print(f"wall clock      : {elapsed:.1f}s  ({elapsed / max(len(tasks), 1):.3f}s/task)")
    print(f"kaggle metric   : {result['score']:.4f}"
          f"  ({result['solved_outputs']}/{result['total_outputs']} test outputs)")
    print(f"format problems : {len(problems)}")
    for p in problems[:10]:
        print("   !", p)

    print("\nrule fired on N tasks (a rule can fire without being right):")
    for n, c in sorted(fired.items(), key=lambda kv: -kv[1]):
        print(f"   {n:<20} {c:>4}")

    solved = [t for t, s in result["per_task"].items() if s > 0]
    print(f"\ntasks with at least one output solved: {len(solved)}")
    if args.verbose and solved:
        print("   " + ", ".join(solved))

    if args.out:
        p = write_submission(submission, args.out)
        print(f"\nwrote {p} ({p.stat().st_size / 1e6:.1f} MB)")
        Path(args.out).with_suffix(".report.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "per_task"}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
