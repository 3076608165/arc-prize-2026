# The submission slot is a noisy instrument

## What ARC Prize 2026 can and cannot resolve, measured by resubmitting identical code

**Tracks:** ARC-AGI-2 and ARC-AGI-3 · **Leaderboard submissions:** ARC-AGI-2 `56204329` (30.97), ARC-AGI-3 `56820929` (28.12) · **Code:** `github.com/3076608165/arc-prize-2026`

## 1. The claim

Every claim on these boards, ours included, rests on an unmeasured assumption: that one submission is one measurement. It is not. Resubmitting byte-identical code shows ARC-AGI-2 resolves about 2.9 points -- fourteen draws of one pinned build span 28.06 to 30.97 (mean 29.48, sd 0.88) -- and that the 1.5-point window holding two thirds of the field is narrower than the instrument. ARC-AGI-3 resolves nothing below about 7 points: four draws of one byte-identical agent-class build returned 28.12, 27.96, 26.03 and 20.98 (mean 25.77, sd 3.33). Four results follow: the reference scorecard aggregates by `max` where it documents a mean, which inverts ablations; the ARC-AGI-2 candidate-selection stage is closed at a measured oracle bound; per-task compute is fixed, so the wall truncates a queue instead of allocating budget; and the strongest public harness-improvement stack transferred negative. Our 30.97 and 28.12 are forked public notebooks. The measurements are ours, and no slot buys them.

## 2. Method

**ARC-AGI-2.** We run the NVARC-lineage public recipe inside Kaggle's 12 h, no-internet envelope: a 4B grid-SFT model, per-task test-time training, 16 augmentation views, and a DFS over decoded candidates, two attempts per scored output. It answers 240 hidden tasks on 4 workers at ~11 min/task, i.e. 11.0 h against an 11.83 h budget. Candidates are copied out once (`repro_nvarc.py --save-candidates`: 1723 files, median 17 per output), so any selector scores against public ground truth offline in seconds at zero quota. Decoding costs ~8 worker-minutes per task; selection is free.

**ARC-AGI-3.** Our own line was the official starter patched to report per-*game* scores and seeded from `sha256(game_id)` with `PYTHONHASHSEED=0`, over the 25 public games, 5 seeds, 3000 steps. It scored 0.27 on the hidden set. The 28.12 above is not that line: it is the top-voted public agent-class build (dfranzen's Milestone-2 solution -- Tufa Labs duck harness driving Qwen3.8-Flash-Next, offline in the same envelope), lifted byte-identical.

**Probes.** Three instruments: *identical-code resubmission*, to measure the board as an instrument; *predeclared thresholds* -- every round writes its reading for each outcome branch before the slot is spent, so a result cannot be reinterpreted; and *matched-subset runs*, the same tasks under a different wall-clock budget, to separate "more time" from "more work".

## 3. Why it works: where the loss actually is

**The ARC-AGI-2 bottleneck is exact completion, not coarse structure.** A depth-3 DSL search reaches the right output *shape* on 42.5% of evaluation tasks and >=90% of cells on 34.2%, with median best-cell accuracy 83.1% -- and solves **0.00%**. The same search solves 3.0% of training tasks, so evaluation is not further from the hypothesis class: it is where exactness is required. ARC scores exact matches only, so widening the hypothesis space moves tasks from *nothing* to *shape-only* to *partial* and is worth exactly zero. The 4B model behaves identically: 32.9% of its outputs land at >=90% of cells correct and score nothing.

**So the mechanism must refine or verify a near-miss, and that route is measurable in advance.** Two attempts per output err in overlapping places: error-set Jaccard median 0.50, and the two agree *and are wrong* on 7.6% of cells. Half of every error is common-mode, so per-cell voting has a ceiling of +4.8 points, not the 2.34x the near-miss share suggests. Measured, voting over merged candidate clusters scores 36/141 against upstream's 42 (`MERGE_TOL=0` reproduces upstream exactly, so the mechanism fails, not the code). Common-mode error means the model holds a wrong hypothesis; more samples of the same distribution cannot repair it. Only a signal from outside the candidate pool -- refitting on n-1 demonstrations and checking the held-out pair -- can.

**Selection is closed, with a bound.** Upstream's `score_full_probmul_3` solves 42/141; the pool's oracle is 51/141. Five mechanisms were built to close that gap and none moved it: per-cell voting -6, demo-inferred shape filter 0, identity demotion 0, best of nine reweightings of the two per-candidate signals -1, cross-model agreement 0. The rivals ranked above the truth differ from it by a median of 5% of cells and are shape-correct; no computable function of `beam_score` and `score_aug` separates them.

**Per-task work is fixed, which is why the wall matters.** The same 24 tasks given ~10x the time produced identical candidates and accuracy (6/34 both; 11.0 vs 10.9 min/task). The deadline truncates the queue, it does not allocate budget: a 240-task submission runs at a 5% margin.

**ARC-AGI-3 scores depth, not breadth, arithmetically.** Per level the metric is `(human_baseline_actions / actions_taken)^2 * 100`, capped, then weighted by level index and capped again at the completed-level share. Two consequences go unstated publicly: level count and score can move in opposite directions, and completing level 1 of an eight-level game caps that game at 2.78 however efficiently it is done. Our exploit-when-known policy cleared more levels and scored less -- the formula punishes it, not the policy. Levels 1-4 at human efficiency cap a game at 27.8, worth 1.16 to a 25-game average: five times our September score.

## 4. Results

| | ARC-AGI-2 | ARC-AGI-3 |
|---|---|---|
| our pair (best two) | **30.97 / 30.56** | **28.12 / 27.96** |
| our rank (2026-11-02) | 934 / 3577 | 1521 / 5043 |
| field median / max | 29.31 / 88.06 | 2.50 / 75.56 |
| honest floor | nine-rule portfolio: 3.07% training, **0.00%** eval | starter random agent: 0.0489 |

**Both boards' identical-code spreads are the result.** ARC-AGI-2 drift is one-directional and does not narrow when the worker-hash seed is pinned, so ~2.9 points is the floor -- and inside the 30.50-31.00 band (241 teams, 1.5 points wide) one point is worth hundreds of ranks. ARC-AGI-3's spread survives deterministic seeding: our September 14x spread was retracted because that agent seeded from `hash()`, and this one does not, yet four draws of the same bytes span 7.1 points.

**A twelve-mechanism harness stack, transferred negative.** The one public ARC-AGI-3 candidate claiming a mechanism rather than a knob added twelve harness fixes to that base (transition and RESET visibility, keep-alive against the 15-minute scorecard close, frame-history serialisation, model-visible text, resume images, solved-level memory, priority flags, edge-band no-ops, sandbox hygiene, start-RESET retry), with public evidence of 31.6 against 25.6 on other game sets. Lifted verbatim it scored **24.80**, below the pair's 26.03 floor. Its predeclared low branch allowed "does not transfer" or "never reached the games"; the full-run log settles it -- 10 of 10 allowlisted games served, 32-275 actions each, 31 levels cleared. The run played the games and scored less anyway. Agent class dominates harness behaviour here by two orders of magnitude: the public agent-class build moved us 0.27 -> 28.12, and the pool's best harness stack moved us -1.2.

## 5. What did not work

All predeclared: depth-3 DSL composition (0.00% eval); augment-8 decoding, predicted to raise the score, returned 27.64 twice, below all five augment-16 samples; per-cell voting as a selector (-6); nine ranker forms (-1, none above baseline); shape filtering and identity demotion (0, 0); a 2B-model ensemble killed at the oracle -- each model held two answers the other did not (+2/34) and every selector collected none; an exploit-when-known ARC-AGI-3 policy that cleared more levels (7.6 vs 6.0) and scored less (0.1356 vs 0.1812); and the harness stack above (24.80).

Two negatives matter most. The forked notebook's claimed 33.89 against our 29.8 +/- 0.8 is not a configuration gap: upstream today is byte-identical to our 2026-09-02 copy, our submission differs only in a budget line and instrumentation, and a bootstrap puts the expected best at 30.97 after 60 more draws while 33.89 sits ~5 sd above our mean -- another platform epoch or an outlier, not a slot. And our own strongest claim went first: the apparent 14x ARC-AGI-3 spread was our `hash()` seeding, voiding every ARC-AGI-3 ablation before 2026-09-10. **Reusable finding:** the reference `Scorecard.score` is documented "the average score of the runs" and returns `max(...)`. Any tuning loop reading it optimises the best single game.

## 6. Related work, and what we would do next

The 2024 and 2025 podiums moved this benchmark with test-time training, verification-heavy inference and ensembles; the recipe we forked is that lineage, and the agent-class ARC-AGI-3 builds are the same story one track over. Public reporting lacks the measurement layer: on ARC-AGI-3 a single-run comparison is noise below ~7 points; on ARC-AGI-2 a change below ~2.9 points is unobservable. Next: a leave-one-demo-out verification signal for ARC-AGI-2, the only lever the oracle bound leaves open, and depth over breadth on ARC-AGI-3, where the level-index weighting makes a first level the cheapest thing to collect.

All numbers, commands and predeclared round files are in `paper/findings.md`; the agents are in `agi2/` and `agi3/`.

<!-- Cover image: figures/draw-distributions.png - identical-code draw distributions for both boards. -->
