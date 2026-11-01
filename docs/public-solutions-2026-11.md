# ARC Prize 2026 — the public pool on 2026-11-01 (T-1 day)

Read at 2026-11-01T10:49Z, one day before the 2026-11-02T23:59Z deadline.

Board snapshot (`kaggle competitions leaderboard -c <slug> -d`, CSV `publicleaderboard-2026-11-01T10:49:05.csv`):

| board | top | our score | our rank | teams |
|---|---|---|---|---|
| ARC-AGI-2 | 88.06 (Tufa Labs) | 30.97 | 934 | 3576 |
| ARC-AGI-3 | 75.56 (Yi-Chia Chen @ Fields Model) | 28.12 | 1487 | 5025 |

The field moved +11.1 (AGI-2) and +56.8 (AGI-3) since the 2026-09-19 read. Nothing in the
open-source pool reaches the top of either board; the question this survey answers is
only "what public artifact is better than ours".

## Platform facts that bound the remaining action

* Scoring takes **24 h+** (`753723`, and Dustin's replies in `747515`); the hosts
  **declined** to extend the selection window (macruzbar, 2026-10-30). A submission made on
  the last day is effectively unscored, so the effective deadline is ~2026-11-01 late.
* One pending submission **blocks** the next one, independently of the 1/day cap.
* Slow-storage wave 2026-10-21/22 zeroed 30-50% of all AGI-3 submissions: model load took
  39-70 min against the harness's 12-min game release and the scorecard's ~15-min
  no-action close (`750924`, `751598`, `751631`). This is the mechanism behind our own
  AGI-3 draws reading 20.98 / 26.03 on the same bytes that read 28.12.

## ARC-AGI-3: the public pool is now a STACK on dfranzen's M2 lift

Every candidate pulled (`kaggle kernels pull <ref> -p experiments/intel-20261031/<name> -m`)
attaches the same pair — datasets `dfranzen/pennyroyal-v253` +
`dfranzen/taaf-kaggle-source-bundle-copy`, models
`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` +
`dfranzen/albucino-qwen3-8-flash-next-drafter` — and ships the **same 498,757-byte
`/kaggle/harness-changes.patch`** (sha256 prefix `07fd859e7ce7` in all four notebooks
inspected). So the differences are all in `bm`/env customization cells, not in the patch.

| notebook | votes | what it adds | claimed/measured |
|---|---|---|---|
| `ondrewsha/arc-agi-3-stack-s16` | 7 | 12 layers: HF (animated≠no-op), **RESET hidden**, KA startup keep-alive (150-min first-request grace), TP (frame history serialised once/frame), MA-text (untrue model-visible text), K3 (no repeated board image on resume), H1 cache-safe solved-level memory, SCH priority flags, MA.4c edge-band no-op fix, HY sandbox hygiene, GW start-RESET retry | "S8/S9 31.6 vs S6/S7 25.6" from hiding RESET alone; scheduler replay +1.0…+2.2 |
| `vickyfym/arc-agi-3-lalalia-final` | 101 | dfranzen M2 + one cell: solved-level memory pinned only at history eviction (cache-safe) | **31.46 LB** in its own header; parent `lalalia9/arc3-franzen-v19` quoted at **35.79 (r40)** |
| `bigbag1983/arc3-franzen-graft-v8` | 0 | D1 env (drain 24576, active streams 8), FR win-path replay (max-over-runs replay of per-level winning segments, zero LLM calls), cache-safe level memory, KP "knowledge assets" prompt, startup heartbeat | composes several authors' *measured* cells |
| `madcat/…-fm2-public-fork-control` | 0 | the control: dfranzen v1 unchanged, 4-pass demo rerun | the 27.80-family control |

Consequence: our live AGI-3 arm is dfranzen **v1 verbatim** (`agi3/out/duck-sota`, 28.12 /
27.96 / 26.03 / 20.98 over four draws). The public pool has since moved ~+3 (lalalia) to
~+7 (stack-s16's own S8/S9 evidence) above it by fixing harness *behaviour*, not by
changing the model. That is a lift, not an invention, and it is the highest-value
remaining action on either board.

## ARC-AGI-2: the pool is still one 4B checkpoint, plus two new legs

All candidates attach `sorokin/qwen3_4b_grids15_sft139` + kernel-source
`sorokin/pip-install-unsloth-flash-patch` on `NvidiaL4` — the same pair our pinned v11
build runs.

* `woguoat/arc2-nvarc-full-cost-sota-triple-pass` (60 votes) — model math unchanged;
  **length-grouped DFS batches up to 8 views** (≈2x faster decode on L4), cheapest-tasks-first
  with a calibrated cost model so every task is processed, leftover time re-runs the least
  certain tasks with fresh seeds and pools under `score_kgmon`, crc32-stable rescoring
  seeds, per-task error isolation. Bounded commit branch (32 min, 4 tasks).
* `yuvrajlamba01/arc-agi2-sota-33-89-plus-cognitive-solver` (23 votes) — the baseline plus
  `arc2_symbolic.py`: demonstration-certified geometric program synthesis filling empty
  attempt slots ("monotone ≥33.89" is its claim, not a measurement). **Its commit branch is
  unbounded (12 h always)** — unusable as a lift without patching the timer.
* `seasthaalores/cprs-arc-agi-2-apex-sota-engine-v7` — same, plus `arc_induction_v2.py` and
  `qwen-lm/qwen2.5-coder/7b-instruct` attached (the verified-induction leg our own
  reference flags as the only second family that addresses the 51/141→42/141 oracle gap).

No public AGI-2 artifact carries a bigger model or a measured score above the 33.89 the
whole family claims, so the board remains "the model class is the only lever", and the
new public work is throughput/coverage (woguoat) plus the symbolic leg (the other two).

## Actions taken off this survey (2026-11-01)

| board | arm | kernel | why |
|---|---|---|---|
| ARC-AGI-3 | `ondrewsha/arc-agi-3-stack-s16` lifted verbatim (bounded commit branch kept: 25-48 min/game, 1 pass, 10 concurrent, 15 games excluded) | `uts3125025522741/arc-agi-3-stack-s16` v1 | ~+3…+7 over our dfranzen-v1 pair, and KA/GW remove the startup-zero failure mode that produced our 20.98/26.03 draws |
| ARC-AGI-2 | `woguoat/arc2-nvarc-full-cost-sota-triple-pass` lifted verbatim (commit branch already bounded to 32 min / 4 tasks) | `uts3125025522741/arc-agi-2-triple-pass` v1 | 2x decode speed + guaranteed full coverage + pooled second run on the uncertain tail; the slot cannot be banked and a redraw has no upside |

Both notebooks are Apache-2.0 / MIT-0 / CC-BY upstream; attribution markdown is kept inside
each lifted notebook, and both are private kernels (publication is a separate decision
gated on `kernelsPublishingDisabledDeadline`, already past at 2026-10-26 23:59Z).

Fired the same day — refs **57870092** (AGI-2, 11:27:06Z) and **57870306** (AGI-3, 11:35:35Z),
both PENDING, both accepted on the first attempt via `competition_submit_code` with the
OUTPUT filename (`submission.json` / `submission.parquet`) — the CLI `-f <local path>` form
returns a bare `400 Bad Request` on these boards.

Commit-smoke readings, recorded next to the arms:

* AGI-2: COMPLETE, 0 tracebacks, `submission.json` with 120 tasks, pass A **and** pass B
  telemetry on the 4 allowlisted tasks, `Reload score 2.5` (= 2.5/4 correct under both
  `score_full_probmul_3` and `score_kgmon`).
* AGI-3: COMPLETE, `benchmark.json` over the 10 allowlisted public games — **10/10 games
  served actions** (the dfranzen-v1 smoke we fired in October served 13 of 25 games **zero**
  actions, which is the mechanism behind our 20.98/26.03 draws), 897,631 generated tokens,
  **31 levels cleared** across the 10 games (sb26 8, re86 5, tr87 5, vc33 3, sc25 3, lf52 2,
  r11l 2, su15 2, bp35 1, sk48 0). The demo games are public and the rerun is scored on the
  private set, so this is a readiness + class signal, not a board prediction.

