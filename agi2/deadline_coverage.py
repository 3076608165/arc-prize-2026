"""Reconstruct observed candidate coverage at the logged worker deadline.

This reads original audit events; it never changes a submitted notebook and
cannot reconstruct the ranking or accuracy of predictions at the deadline.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_number(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def summarize_deadline(directory: Path, expected_output_ids=None):
    files = sorted(directory.glob("rank-*.jsonl"))
    records, errors = [], []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            location = f"{path.name}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{location}: invalid or truncated event")
                continue
            if not isinstance(record, dict) or not isinstance(record.get("event"), str):
                errors.append(f"{location}: invalid event object")
                continue
            rank = record.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int):
                errors.append(f"{location}: missing or invalid worker rank")
                continue
            records.append(record)

    starts = {}
    for event in records:
        if event["event"] == "worker_started":
            starts.setdefault(event["rank"], []).append(event.get("deadline"))
    deadlines = {}
    for rank, values in starts.items():
        if len(values) != 1 or not finite_number(values[0]):
            errors.append(f"worker {rank}: missing, invalid, or duplicated deadline")
        else:
            deadlines[rank] = values[0]
    distinct_deadlines = set(deadlines.values())
    if len(distinct_deadlines) > 1:
        errors.append("Workers have different deadlines; there is no single common cutoff")

    before, after, unknown, observed = set(), set(), set(), set()
    first_candidate, last_event, finished_after = {}, {}, set()
    unknown_workers = set()
    for event in records:
        rank, name, when = event["rank"], event["event"], event.get("time")
        deadline = deadlines.get(rank)
        if rank >= 0 and finite_number(when):
            last_event[rank] = max(last_event.get(rank, when), when)
        if name == "task_finished" and deadline is not None and finite_number(when) and when > deadline:
            if isinstance(event.get("task"), str):
                finished_after.add(event["task"])
        if name != "candidates_written":
            continue
        output = event.get("output_id")
        count = event.get("count")
        if not isinstance(output, str) or not output or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            errors.append(f"worker {rank}: invalid candidate-write event")
            continue
        observed.add(output)
        if finite_number(when):
            first_candidate[output] = min(first_candidate.get(output, when), when)
        if deadline is None or not finite_number(when):
            unknown.add(output)
            if deadline is None:
                unknown_workers.add(rank)
            if not finite_number(when):
                errors.append(f"{output}: missing or invalid candidate-write timestamp")
        elif when <= deadline:
            before.add(output)
        else:
            after.add(output)

    expected = set(expected_output_ids) if expected_output_ids is not None else observed
    missing = expected - observed
    extra = observed - expected
    if missing:
        errors.append("Summary candidates are missing from original candidate-write events")
    if extra:
        errors.append("Original candidate-write events contain candidates absent from the summary")
    unknown.update(missing)
    # A known timely write proves timely coverage even if a later write has
    # unknown timing. Unknown earlier writes prevent claiming 'first seen late'.
    unresolved = unknown - before
    late_only = after - before - unknown
    tail_by_worker = {str(rank): max(0.0, when - deadlines[rank])
                      for rank, when in sorted(last_event.items()) if rank in deadlines}
    complete = bool(files and deadlines) and not errors and not unknown_workers and not unresolved
    return {
        "status": "complete" if complete else "incomplete" if files else "unavailable",
        "deadline_unix": next(iter(distinct_deadlines)) if len(distinct_deadlines) == 1 else None,
        "worker_deadlines_unix": {str(k): v for k, v in sorted(deadlines.items())},
        "outputs_confirmed_by_deadline": sorted(before),
        "outputs_first_observed_after_deadline": sorted(late_only),
        "outputs_with_any_late_write": sorted(after),
        "outputs_with_unknown_timing": sorted(unresolved),
        "outputs_missing_from_raw_events": sorted(missing),
        "raw_outputs_absent_from_summary": sorted(extra),
        "workers_missing_unambiguous_deadline": sorted(unknown_workers),
        "first_observed_candidate_time_unix": dict(sorted(first_candidate.items())),
        "counts": {"confirmed_by_deadline": len(before), "first_observed_after_deadline": len(late_only),
                   "any_late_write": len(after), "unknown_timing": len(unresolved),
                   "all_observed_candidate_outputs": len(observed), "tasks_finished_after_deadline": len(finished_after)},
        "observed_tail_seconds_by_worker": tail_by_worker,
        "maximum_observed_tail_seconds": max(tail_by_worker.values(), default=None),
        "errors": errors,
        "interpretation": "Coverage uses logged candidate-write time <= the worker's logged deadline. Missing or ambiguous deadlines remain unknown. Writes are logged after file close, so a write near the boundary may have actually finished slightly earlier. Tail time ends at the last observed worker event and is a lower bound on process shutdown time. Final submission accuracy may include later candidates and cannot be described as accuracy at this cutoff.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Original coverage_audit directory")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = summarize_deadline(args.directory)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
