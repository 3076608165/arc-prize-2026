# What a submission slot can resolve on ARC Prize 2026

## Resolution limits, a mis-documented scorer, and a closed selection stage

**Track:** ARC-AGI-2 / ARC-AGI-3 · **Submissions:** ARC-AGI-2 `56204329` (public 30.97),
ARC-AGI-3 `56268909` (public 0.26)

## 1. The claim

Every result on these two boards rests on an unmeasured assumption: that one
submission is one measurement. It is not. Resubmitting **byte-identical** code
shows ARC-AGI-2 resolves about 2 points and ARC-AGI-3 resolves nothing at all
until the agent's own seed is pinned — at which point it becomes exactly
deterministic and the seed, not the board, moves the score about 9x. Three further
measurements follow from taking that seriously: the reference implementation's
aggregation function is documented as a mean and implemented as a max, which
inverts ablations; the ARC-AGI-2 candidate-selection stage is closed, with a
measured oracle bound; and per-task compute is fixed rather than elastic, so the
12-hour wall truncates a queue instead of allocating budget. The headline score
is not the contribution — 30.97 of 73.33 is a forked public notebook. The
measurements are.

## 2. Method

**ARC-AGI-2.** We run the NVARC-lineage public recipe inside Kaggle's 12 h,
no-internet envelope: a 4B model, per-task test-time training, 16 augmentation
views per task, and a DFS over decoded candidates, with two attempts emitted per
scored output. It answers 240 hidden tasks on 4 workers at ~11 min/task, i.e.
11.0 h against the 11.83 h budget. Candidates are also copied out of the kernel
once (`repro_nvarc.py --save-candidates`: 1723 files, median 17 candidates per
output), so any selector can be scored against public ground truth offline in
seconds at zero quota. Decoding costs ~8 worker-minutes per task; selection is
free. Only the free half is ours.

**Probes.** Three instruments are used throughout, each cheap and each reused:

1. *Identical-code resubmission* to measure the board as an instrument.
2. *Predeclared thresholds*: every round writes its reading, per outcome branch,
   into a round file before the slot is spent, so a result cannot be reinterpreted
   afterwards.
3. *Matched-subset runs*: rerun the same task subset under a different wall-clock
   budget to separate "more time" from "more work".

**ARC-AGI-3.** We use the official starter, patched to report the per-*game*
scores the engine already computes, and to seed each agent from
`sha256(game_id)` with `PYTHONHASHSEED=0` so that a seed means the same thing in
every process. Local arms are run over the 25 public games, 5 seeds, 3000 steps.

## 3. Why it works: where the loss actually is

**The ARC-AGI-2 bottleneck is exact completion, not coarse structure.** A
depth-3 DSL search reaches the right output *shape* on 42.5% of evaluation tasks
and ≥90% of cells on 34.2%, with median best-cell accuracy 83.1% (p90 95.7%) —
and solves **0.00%**. The same distribution holds on training tasks, where that
search does solve 3.0%: evaluation is not further from the hypothesis class, it
is where exactness is required. ARC scores exact matches only, so widening the
global hypothesis space moves tasks from *nothing* to *shape-only* to *partial*
and is worth exactly zero. The same holds for the 4B model: 32.9% of its outputs
land at ≥90% of cells correct and score nothing; median cell accuracy over all
outputs is 94.1%.

**So the mechanism must refine or verify a near-miss, and the obvious way to do
that is measurable in advance.** Two attempts are emitted per output, and they
err in overlapping places: error-set Jaccard median 0.50, and the two attempts
agree **and are wrong** on 7.6% of cells. Half of every error is common-mode, so
per-cell voting has a realistic ceiling of +4.8 points, not the 2.34x that the
near-miss share suggests. Measured, per-cell voting over merged candidate
clusters scores **36/141 against upstream's 42** (`MERGE_TOL=0` reproduces
upstream exactly, so this is the mechanism failing, not the code). Common-mode
error means the model holds a wrong hypothesis; more samples of the same
distribution cannot repair it. Only a signal from outside the candidate pool —
re-fitting on n−1 demonstrations and checking the held-out pair — can.

**Selection is closed, with a bound.** Upstream's `score_full_probmul_3` solves
42/141; the pool's oracle is 51/141. Five mechanisms were built to close that
gap and none moved it: per-cell voting −6, demo-inferred shape filter 0,
identity-demotion 0, best of nine reweightings of the two available per-candidate
signals −1, cross-model agreement 0. The rivals ranked above the truth differ
from it by a **median of 5% of cells** and are shape-correct; no computable
function of `beam_score` and `score_aug` separates them.

**Per-task work is fixed, and that is why the wall matters.** Giving the same 24
tasks ~10x the time per task produced identical candidates and identical
accuracy (6/34 both) — 11.0 min/task against 10.9. The global deadline truncates
the queue; it does not allocate budget. A 240-task submission runs at a ~5%
margin, which is presumably where upstream tuned it.

## 4. Results

| | ARC-AGI-2 | ARC-AGI-3 |
|---|---|---|
| our best public | **30.97** | **0.26** |
| field median / max (2026-09-04) | 27.64 / 73.33 | 0.27 / 7.51 |
| seeded-code reruns | 29.31→30.97 (1.66 pts) | 0.03, 0.03 (0.00) |
| honest floor | 9-rule portfolio: 3.07% training, **0.00%** eval | starter random agent: 0.0489 |

The floor is ours and it is unimpressive on purpose: nine hand-written global
rules score zero on evaluation, and adding compositional search lifts training by
57% relative and evaluation by nothing. Rule portfolios are not a route to this
benchmark. Board precision: five byte-identical resubmits of the original
notebook spanned 1.67 points (30.14, 29.72, 28.47, 28.47, 28.89) and the drift
was one-directional; pinning our worker-hash seed did **not** narrow it (three
reruns: 29.31, 30.56, 30.97). The residual is platform-side bf16 and wall-clock
nondeterminism, so ~2 points is the final floor. On the hidden ARC-AGI-3 set the
metric is discrete: one extra level cleared in one game is the whole distance
between 0.03 and 0.26. Five seeds of one unchanged agent, differing in nothing
but the seed, returned 0.03, 0.21, 0.06, 0.04 and 0.26.

## 5. What did not work

Negative results, all predeclared and reproducible: depth-3 DSL composition
(0.00% eval); augment-8 decoding, which was predicted to *raise* the score and
returned 27.64 twice, below every one of five augment-16 samples; per-cell
voting as a selector (−6); nine ranker forms (−1, none above baseline); shape
filtering and identity demotion (0, 0); a 2B-model ensemble, killed at the
oracle — each model held two answers the other did not (+2/34 ceiling), and
every selector collected **none** of it; an exploit-when-known policy on
ARC-AGI-3 that cleared *more* levels (7.6 vs 6.0) and scored *less* (0.1356 vs
0.1812).

**We also retract our own strongest claim.** An apparent 14x identical-code
spread on ARC-AGI-3 (0.01…0.18) was not the board: our agent seeded from
`hash(game_id)`, which Python randomises per process. Pinned, the board
reproduces to two decimals — and the variance did not vanish, it moved into the
seed. Every ARC-AGI-3 ablation in our project before 2026-09-10 is consequently
void, including one that reverted a mechanism on broken randomisation.
**Reusable finding:** the reference `Scorecard.score` is documented "the average
score of the runs" and returns `max(...)`. Any tuning loop reading it is
optimising the best single game.

## 6. Related work, and what we would do next

The 2024 and 2025 podiums moved the benchmark with test-time training,
verification-heavy inference, and ensembles; the public recipe we forked is that
lineage. What is missing from the public reporting is the measurement layer this
writeup adds, and it changes what a result means: on ARC-AGI-3 a single-run
comparison is noise unless the agent seeds deterministically, and on ARC-AGI-2 a
change below ~2 points is not observable in one submission. Next: a
leave-one-demo-out verification signal for ARC-AGI-2 (the only lever the oracle
bound leaves open), and depth over breadth on ARC-AGI-3 — clearing levels 1–4 of
one eight-level game at human efficiency caps that game at 27.8 of 100, worth
1.16 to the game-averaged score, **5x our entire current score** — because the
level-index weighting makes a first level the cheapest thing to collect.

All numbers, commands and predeclared round files are in
[findings.md](findings.md); the shipped agent is in `agi2/` and `agi3/`.

<!-- Cover image: agi2/arclib/viz.py task_to_html (demos, truth, both attempts). -->
