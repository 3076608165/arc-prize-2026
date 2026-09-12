"""Instrument NVARC work coverage without reading evaluation ground truth."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def audit(rank, task, event, **details):
    directory = Path(os.environ.get("ARC_COVERAGE_DIR", "/kaggle/working/coverage_audit"))
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(rank=rank, task=task, event=event, time=time.time(), **details)
    with (directory / f"rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def summarize(directory: Path) -> dict:
    records, errors = [], []
    for path in sorted(directory.glob("rank-*.jsonl")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(f"{path.name}:{number}: invalid or truncated event")
    records.sort(key=lambda r: r["time"])
    tasks, active, worker_errors = {}, {}, []
    for event in records:
        rank, task, name = event["rank"], event["task"], event["event"]
        if name == "worker_error":
            worker_errors.append(event)
            task = active.get(rank)
        if task is None:
            continue
        row = tasks.setdefault(task, {"task_id": task, "events": [], "outputs": set(),
                                      "candidate_count": 0, "batch_completed": 0})
        row["events"].append(event)
        if name == "task_started":
            active[rank] = task
        if name == "task_finished":
            active.pop(rank, None)
        if name == "queued":
            row["expected_outputs"] = event["expected_outputs"]
        elif name == "batches_planned":
            row["batch_planned"] = event["count"]
        elif name == "batch_completed":
            row["batch_completed"] += 1
        elif name == "candidates_written":
            row["outputs"].add(event["output_id"])
            row["candidate_count"] += event["count"]
    for row in tasks.values():
        events = row.pop("events")
        names = {e["event"] for e in events}
        row["started"] = "task_started" in names
        row["training_started"] = "training_started" in names
        row["training_completed"] = "training_completed" in names
        row["finished"] = "task_finished" in names
        row["truncation_reasons"] = sorted({e["reason"] for e in events if e["event"] == "task_truncated"})
        capped = any(e.get("capped", False) for e in events if e["event"] == "dfs_returned")
        row["dfs_cap_observed"] = capped
        row["error"] = "worker_error" in names
        row["decoding_completed_without_observed_cap"] = bool(
            row["finished"] and row.get("batch_planned", 0) > 0
            and row["batch_completed"] == row["batch_planned"]
            and not capped and not row["truncation_reasons"] and not row["error"])
        started = next((e["time"] for e in events if e["event"] == "task_started"), None)
        finished = next((e["time"] for e in events if e["event"] == "task_finished"), None)
        row["elapsed_seconds"] = finished - started if started is not None and finished is not None else None
        row["candidate_output_ids"] = sorted(row.pop("outputs"))
        row["outputs_with_candidates"] = len(row["candidate_output_ids"])
        expected_ids = {f"{row['task_id']}_{i}" for i in range(row.get("expected_outputs", 0))}
        row["all_outputs_have_candidates"] = bool(expected_ids) and set(row["candidate_output_ids"]) == expected_ids
        if not set(row["candidate_output_ids"]) <= expected_ids:
            errors.append(f"{row['task_id']}: unexpected candidate output ID")
        row["started_but_incomplete"] = row["started"] and not row["finished"]
        row["no_valid_candidates_after_finished"] = row["decoding_completed_without_observed_cap"] and row["outputs_with_candidates"] == 0
        row["past_global_deadline_at_finish"] = any(e.get("past_deadline", False) for e in events if e["event"] == "task_finished")
        if sum(e["event"] == "queued" for e in events) != 1:
            errors.append(f"{row['task_id']}: missing or duplicate queue event; do not combine multiple runs")
    counts = {key: sum(bool(row.get(key)) for row in tasks.values()) for key in (
        "started", "training_started", "training_completed", "finished", "decoding_completed_without_observed_cap",
        "no_valid_candidates_after_finished", "past_global_deadline_at_finish",
        "all_outputs_have_candidates", "started_but_incomplete", "error", "dfs_cap_observed")}
    counts.update(queued=sum(e["event"] == "queued" for e in records),
                  not_started=sum(not r["started"] for r in tasks.values()),
                  outputs_with_candidates=sum(r["outputs_with_candidates"] for r in tasks.values()),
                  expected_outputs=sum(r.get("expected_outputs", 0) for r in tasks.values()))
    return {"counts": counts, "tasks": tasks, "worker_errors": worker_errors,
            "worker_deadline_stops": sum(e["event"] == "worker_deadline_stop" for e in records),
            "event_errors": errors, "empty_audit": not records,
            "interpretation": "Stage completion and candidate coverage only; no correctness measurement. Missing events mean unknown/incomplete. Filler grids never count as candidates. DFS caps are conservatively inferred from elapsed time/deadline."}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Expected one instrumentation anchor: {old[:80]!r}")
    return source.replace(old, new, 1)


def patch_notebook(notebook):
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.startswith("%%writefile arc_solver.py\n"):
            source = source.replace("%%writefile arc_solver.py\n", "%%writefile arc_solver.py\nfrom coverage_audit import audit as _audit\n", 1)
            for old, extra in [
                ("def worker(rank, queue, end_time):", '\n    _audit(rank, None, "worker_started", deadline=end_time)'),
                ("        start_time = time.time()", '\n        _audit(rank, key, "task_started")'),
                ("            stats = trainer.train()", '\n            _audit(rank, key, "training_completed")'),
                ("        with torch.inference_mode():", '\n            _audit(rank, key, "batches_planned", count=len(batches))'),
                ('            print(f"[Rank {rank}] stop!")', '\n            _audit(rank, None, "worker_deadline_stop")'),
                ('                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")', '\n                    _audit(rank, key, "task_truncated", reason="global_deadline" if time.time() > end_time else "per_task_budget")'),
                ('                            pickle.dump(decoded_result, f)', '\n                        _audit(rank, key, "candidates_written", output_id=bk, count=len(decoded_result))'),
                ('        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")', '\n        _audit(rank, key, "task_finished", elapsed=spend_time, past_deadline=time.time() > end_time)'),
            ]:
                source = replace_once(source, old, old + extra)
            source = replace_once(source, "            stats = trainer.train()", '            _audit(rank, key, "training_started")\n            stats = trainer.train()')
            call = "                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)"
            source = replace_once(source, call, '                _audit(rank, key, "decode_batch_started", subkeys=list(subkeys))\n                _dfs_start = time.time()\n' + call + '\n                _audit(rank, key, "dfs_returned", elapsed=time.time()-_dfs_start, capped=time.time()-_dfs_start >= 540 or time.time() >= end_time)')
            anchor = '        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2\n        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")'
            source = replace_once(source, anchor, '                _audit(rank, key, "batch_completed")\n\n' + anchor)
            compile(source.split("\n", 1)[1], "arc_solver.py", "exec")
            found.add("solver")
        elif source.startswith("%%writefile starter.py\n"):
            source = source.replace("%%writefile starter.py\n", "%%writefile starter.py\nfrom coverage_audit import audit as _audit\n", 1)
            source = replace_once(source, "        queue.put(key)", '        _audit(-1, key, "queued", expected_outputs=len(data[key]["test"]))\n        queue.put(key)')
            source = replace_once(source, "    worker(rank, queue, end_time)", '    try:\n        worker(rank, queue, end_time)\n    except BaseException as exc:\n        _audit(rank, None, "worker_error", error_type=type(exc).__name__)\n        raise')
            compile(source.split("\n", 1)[1], "starter.py", "exec")
            found.add("starter")
        cell["source"] = source.splitlines(keepends=True)
    if found != {"solver", "starter"}:
        raise ValueError("Missing expected NVARC modules")
    notebook["cells"].insert(1, {"cell_type": "code", "metadata": {}, "outputs": [],
        "execution_count": None, "source": ("%%writefile coverage_audit.py\n" + Path(__file__).read_text(encoding="utf-8")).splitlines(keepends=True)})
    notebook["cells"].append({"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
        "source": ['from coverage_audit import summarize\n', 'from pathlib import Path\n', 'import json\n',
                   '_audit_summary = summarize(Path("/kaggle/working/coverage_audit"))\n',
                   'Path("/kaggle/working/coverage_summary.json").write_text(json.dumps(_audit_summary, indent=2), encoding="utf-8")\n',
                   'print(_audit_summary["counts"])\n']})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["counts"], indent=2))
