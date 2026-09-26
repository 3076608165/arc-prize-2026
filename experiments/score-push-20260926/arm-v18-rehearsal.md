# Kernel v18 (twopass-largest-first) rehearsal — the 2026-09-27 fire gate

**Read at 2026-09-26T13:29Z (COMPLETE), output downloaded to
`agi2/out/twopass-lf-v18-run/`. Gate: GREEN — the arm is safe to fire for
2026-09-27.** It spent 2h09m QUEUED before it ever ran (GPU session queue), which is
why the earlier watch — 90 minutes at 60 s — read nothing but QUEUED and this file
stood at "unread" for a while: unread meant *not run yet*, not *broken*.

| check | reading |
|---|---|
| status | `KernelWorkerStatus.COMPLETE` at 13:29:19Z (QUEUED 09:20Z → 12:44Z, then RUNNING) |
| traceback in log | 0 matches |
| coverage counts | `'error': 0`, `outputs_with_candidates: 5 / expected_outputs: 5`, `started 4 / finished 4 / not_started 0`, `queued: 8` (the 4-task allowlist doubled by the second pass) |
| `submission.json` | written, 23,253 bytes (same size as v17's rehearsal) |
| log wall clock | 2,537.7 s = **42.3 min** (v17 42.7 min, marker build 28.5 min) |
| identity | every `%%writefile` cell of `agi2/out/twopass-lf-v18/submission.ipynb` matches the file the run left behind, modulo the trailing newline the magic adds (rstrip-equal for all five: coverage_audit, arc_loader, arc_decoder, arc_solver, starter) |
| the one knob | the largest-first key (`len(g) * len(g[0])`) is present in the run's emitted `starter.py` — the queue reorder really shipped, this is not a rebuild of v17 |
| pass-id wiring | `_pass_store` present in the emitted `arc_solver.py`, `__PASS2__` absent |
| preflight | `*** Reload score: 2.5` on the 4-task allowlist — same as v17 and the base build, against 3.0 for the marker (v15) build |
| local tests | `agi2/test_two_pass.py` 9/9 pass, including `order_largest_first` |
| build flags | `--two-pass --largest-first --audit-coverage --worker-hash-seed 0 --hours 11.83` (v17's flags plus largest-first) |

## What this arm is testing, and what it is not

Scheduling only: pass B queues behind every pass-A task, and with the expensive
puzzles first the second pass now reaches the tail the plain two-pass arm never got
to. The candidate pool is unchanged, so a gain would say the earlier nulls (v15
30.56; v17 pending) were coverage rather than the idea; a null here retires the
second decode round on this family.

## The risk trade that was recorded while this was unread

The arm stayed in `scripts/arm-queue.json` as version 18 before this read, because a
pure-scheduling reorder of an already-green build shares its worst case (a lost
slot — an unverified run that dies before writing `submission.json` cannot be
submitted, exactly like `-v None` retrying to its deadline) with not firing at all.
That is now moot: the rehearsal ran and is green, so the 2026-09-27 fire submits a
build whose own log shows the new code path executing and writing `submission.json`.
Had it read red, the arm would have been removed from the queue and the day's slot
held deliberately instead.
