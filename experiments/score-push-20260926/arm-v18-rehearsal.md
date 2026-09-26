# Kernel v18 (twopass-largest-first) rehearsal — the 2026-09-27 fire gate

Status at 2026-09-26T10:53Z (20:53 local): **NOT READ — the version is still
`KernelWorkerStatus.QUEUED`**, 93 minutes after the push. The watcher polled it
every 60 s for 90 minutes and saw nothing but QUEUED; the GPU session queue is the
gate, not the build.

| check | reading |
|---|---|
| pushed | kernel version 18, 2026-09-26T09:20Z (one knob vs v17: cell 7's `_queue_keys` only — verified by cell-by-cell diff of the two builds) |
| status | `KernelWorkerStatus.QUEUED` at 10:53Z, no run yet |
| build flags | `--two-pass --largest-first --audit-coverage --worker-hash-seed 0 --hours 11.83` (identical to v17's flags but for largest-first) |
| local tests | `agi2/test_two_pass.py` 9/9 pass, including `order_largest_first` |
| traceback / coverage / `submission.json` | unknown until the run exists |

## Decision recorded with the queue advance

The arm is left in `scripts/arm-queue.json` as version 18, so the 2026-09-27 fire
has valid bytes. Reason, stated as the trade it is: this arm is a pure-scheduling
reorder of an already-green build (v17's rehearsal was COMPLETE, no traceback, 5/5
outputs with candidates), and the failure mode of firing an unverified build — the
run dies before writing `submission.json` and the version cannot be submitted — is
the *same lost slot* as leaving the version null (`fire_draws.py` would send
`-v None`, which fails and retries to its deadline, never submitting anything). So
the choice is the same worst case with the arm's upside attached, not a risk taken
for nothing.

What must NOT happen: the rehearsal read as red (traceback, 0 outputs with
candidates, no `submission.json`) and the arm fired anyway. Read this gate in the
next session — one command — before 10:05 local on 2026-09-27:

    kaggle kernels output uts3125025522741/arc-agi-2-nvarc-repro -p agi2/out/twopass-lf-v18-run
    # then: traceback count, 'error': 0, outputs_with_candidates 5/5, submission.json written,
    #       emitted files byte-identical to agi2/out/twopass-lf-v18/submission.ipynb cells

If red: remove this arm from `scripts/arm-queue.json` (the queue then selects
`twopass-fix`, which is already fired, i.e. nothing fires) and hold the day's slot
deliberately rather than spend it on bytes that cannot produce a submission.
