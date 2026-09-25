# Kernel v17 (twopass-item-passid) rehearsal — the 2026-09-26 fire gate

Read **before** the slot is spent, which is the rule the 9/24 round paid a day to learn.
Source: newest version of `uts3125025522741/arc-agi-2-nvarc-repro`, i.e. v17
(`kaggle kernels output`, downloaded 2026-09-25 12:09 local).

| check | reading |
|---|---|
| status | `KernelWorkerStatus.COMPLETE` at 02:09:34Z |
| traceback in log | none (0 matches) |
| coverage counts | `'error': 0`, `outputs_with_candidates: 5 / expected_outputs: 5`, `started 4 / finished 4 / not_started 0` |
| `submission.json` | written, 23,253 bytes |
| log run time | 2,561 s (42.7 min) vs 1,707 s (28.5 min) for the marker build — the wiring does change the work done |
| identity | run's `arc_solver.py` / `starter.py` / `arc_decoder.py` byte-identical to the `%%writefile` cells of `agi2/out/twopass-itemid/submission.ipynb` (the local `arc_solver_emitted.py` differs only by CRLF) |
| pass-id wiring | `_pass_store` present in the emitted worker; `__PASS2__` absent |

Verdict: **submittable.** The arm is safe to fire for 2026-09-26.

## Preflight reading, recorded for what it is

`*** Reload score: 2.5` on the 4-task commit allowlist, against 3.0 for the marker
version (v15), 2.5 for the orig-numerics control (v16) and 2.5 for the base build.
That is one task out of four and is not a score — but it is the opposite direction from
the arm's hypothesis, so it is written down before the board can be read as agreement:
the fully-applied pass B did not reproduce one task the partial (marker) version got.

## What this arm is actually testing

The 9/25 two-pass arm (v15) returned its predeclared null, 30.56, with only one of four
workers decoding a genuinely decorrelated pass B. v17 is the version where the pass id
travels with each queued item, so all four workers decode pass B into its own store and
the pooled selection sees the design as intended. Predeclared: the same null falsifies
the second decode round as a lever on this family; a gain means the marker version's
partial coverage, not the idea, was what read flat.
