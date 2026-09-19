# Measured findings

Everything here is reproducible from this repo with the command shown. These
are the raw material for the Paper Track writeup — particularly the Theory and
Progress criteria, which reward explaining *why* something works and reporting
what did not.

---

## Leaderboard, 2026-09-04

| competition | first submission | now | rank | field median | field max |
|---|---|---|---|---|---|
| ARC-AGI-2 | 0.00 (1760/1760) | **30.14** | **591 / 1802** | 27.64 | 73.33 |
| ARC-AGI-3 | 0.15 | 0.15 (best kept) | 2006 / 2773 | 0.27 | 7.51 |

### ARC-AGI-2: switching families was the whole thing

0.00 to 30.14, last place to rank 591 of 1802, above the field median. Nothing
about our own ideas produced that -- it is the NVARC-lineage public notebook,
forked and run. The lesson is the one the 0.00 already implied and this
confirms: on this benchmark the hypothesis class *is* the result. Depth-3 DSL
search scored zero; a 4B model with per-task test-time training scores 30 with
no cleverness from us at all.

It also cost almost nothing in quota -- ~0.9h for the commit, with the 240-task
work running on Kaggle's side during the rerun.

### Leaderboard precision, measured on both tracks

The same experiment on both boards -- resubmit byte-identical code and see what
comes back -- gives opposite answers, and settles where effort belongs.

| track | identical-code resubmits | spread | smallest change the board can resolve |
|---|---|---|---|
| ARC-AGI-2 | 30.14, 29.72, 28.47 | **1.67 pts (5.6%)** | ~2 points |
| ARC-AGI-3 | 0.15, 0.04, 0.12, 0.01, 0.14 | **14x** | more than an order of magnitude |

**ARC-AGI-2 is a precise instrument** -- but less precise than the first two
samples suggested. At n=2 the spread looked like 1.4%; the third resubmit came
back 28.47 and widened it to 5.6%. The honest floor is **~2 points**, not one.
Still two orders of magnitude better than the other board.

**ARC-AGI-3 is not a measuring device at all** at anything below an order of
magnitude. Our best is 0.15; demonstrating a real improvement would require
roughly 2.0, which sits between the field median (0.27) and the leader (7.51).
Every ARC-AGI-3 comparison we have made, and every published single-run
comparison by anyone else, is inside that noise.

That is the resource decision: **ARC-AGI-2 gets the effort.** ARC-AGI-3 keeps
its daily submission because the slot is free, but no further engineering until
something with an order-of-magnitude story exists.

### ARC-AGI-3 detail: the leaderboard is noise at our effect sizes

| version | local true score (5 seeds) | leaderboard |
|---|---|---|
| v1, argmax policy | 0.1141 | 0.15 |
| v2, sampling policy | 0.1812 | **0.04** |
| **v2 again, byte-identical resubmit** | -- | **0.12** |

The third row settles it. The *same kernel version*, resubmitted unchanged,
returned 0.04 and then 0.12 -- a **3x spread on identical code**. So the
apparent "sampling is 73% worse" was never a fact about the policy; 0.04 was a
low sample of a wide distribution, and 0.15 / 0.04 / 0.12 are mutually
indistinguishable.

This exonerates the local metric rather than vindicating it: nothing has
contradicted it, and nothing has confirmed it either. It also matches the local
picture exactly -- our own 5-seed spread for one policy was [0.1179, 0.3055],
also ~2.6x.

**The operational consequence stands and is now measured, not inferred.** At
one submission per day and ~3x single-run variance, no ARC-AGI-3 change below
roughly 3x is confirmable on the board. Chasing small gains there is spending
days to sample noise. Either pursue only large effects -- which the cap
structure says means depth within a game, not more entrances -- or treat the
local 5-seed mean as the target and stop paying submissions to check it.

Worth stating for the writeup: **a benchmark whose leaderboard cannot resolve a
1.5x improvement is a benchmark where published single-run comparisons should
be read with suspicion**, including everyone else's.

---

## ARC-AGI-2: the classical search family gets close and never closes

`python agi2/run_baseline.py --split {evaluation,training} [--full]`

| pipeline | training (1000) | evaluation (120) |
|---|---|---|
| 9 hand-written global rules | 3.07% (33/1076) | **0.00%** (0/167) |
| + depth-3 DSL composition, symmetry repair, 3×3 neighbourhood rule | 4.83% (52/1076) | **0.00%** (0/167) |

Adding compositional search lifted training by 57% relative and evaluation by
nothing at all. Zero is a suspicious number, so the next experiment asks how
close the search actually gets.

### How close is "zero"?

`python agi2/analyze.py --split {evaluation,training}`

For each task, the best any program in the depth-3 space achieves on the
demonstration pairs:

| | evaluation (120) | training (300) |
|---|---|---|
| solved exactly | **0.0%** | 3.0% |
| partial (right shape, ≥90% cells) | 34.2% | 28.3% |
| shape-only (right shape, <90% cells) | 42.5% | 49.0% |
| nothing (cannot even match output shape) | 23.3% | 19.7% |
| median best cell accuracy | **83.1%** | 83.3% |
| p90 best cell accuracy | **95.7%** | 96.5% |

**The two distributions are the same everywhere except the one row that
scores.** Evaluation is not further away from the hypothesis class — the search
lands just as close on it (median 83% of cells correct, p90 96%) and then
never closes the gap.

### Why this matters

ARC scores exact matches only. A grid with 95.7% of its cells right is worth
precisely as much as a blank one. So on this benchmark:

- The bottleneck is **not** coarse structure. Something in the search space
  already reproduces the grid's size, its layout, and the large majority of its
  contents on a third of eval tasks.
- The bottleneck is the **last few percent of cells** — the exceptions, the
  one object that behaves differently, the rule that has a special case.
- Therefore *widening* the global hypothesis space is the wrong lever: it moves
  tasks from `nothing` to `shape-only` and from `shape-only` to `partial`,
  which is worth zero. A method that **refines a near-miss** — proposes,
  checks against the demos, and repairs the cells that disagree — attacks the
  gap that actually costs points.

That is a falsifiable claim and it suggests its own ablation: hold the
proposal stage fixed, add and remove the repair stage, and read off how many
tasks cross from 96% to 100%.

### Task-shape statistics that shaped the design

`evaluation` vs `training`, from `arclib.loader`:

| | evaluation | training |
|---|---|---|
| shape-preserving tasks | 81/120 (68%) | 680/1000 (68%) |
| median input cells | **299** | 100 |
| median colors per input | 5 | 3 |
| tasks with 2+ test inputs | 45/120 | 69/1000 |

Eval grids are ~3× larger and carry more colors than training grids, and 120
tasks yield 167 scored outputs. Any per-task compute budget has to be planned
against 167, not 120.

---

## Two more closures, 2026-09-13

**The residual ARC-AGI-2 variance is Kaggle's, not ours.** v11 (worker hash
seed fixed) resubmitted three times: 29.31, 30.56, 30.97 -- a 1.66-point
range, indistinguishable from the mis-seeded v4's 1.67 over five. Fixing our
seeding removed nothing. The remaining spread is GPU nondeterminism and
wall-clock, which we cannot touch. The ~2-point resolution floor is final.
(30.97 is also a new best.)

**The public 25 games do not predict the hidden games -- not in level, not in
order.** Same deterministic agent, two seeds:

| seed | local (25 public, 5000 cap) | board |
|---|---|---|
| 0 | **0.305** | 0.03 |
| 2 | 0.147 | **0.06** |

Absolute gap 10x; ordering reversed. With the board now known to be
deterministic, this is not noise -- it is distribution shift between the public
and hidden game sets. The local ARC-AGI-3 harness, which was worth building to
find the scorer bug and the seeding bug, is not a proxy for the board. At n=2
this is suggestive; today's seed-1 submission (local worst) makes it n=3.

The consequence, if it holds: ARC-AGI-3 improvement can only be validated on
the board, which is deterministic but yields exactly one evaluation per day
with no local prediction of it. That is a very slow hill-climb on a function
we cannot see.

## REVERSAL: the ARC-AGI-3 leaderboard is deterministic; the 14x was ours

The variance probe of 2026-09-12 resubmitted byte-identical *seeded* code on
both boards, with the readings declared in advance in
`experiments/score-push-20260912/round.json`.

| track | seeded version | sample 1 | sample 2 | spread | predeclared reading |
|---|---|---|---|---|---|
| ARC-AGI-3 | v4 (sha256 game seed) | 0.03 | **0.03** | **0.00** | within 2x -> our RNG dominated |
| ARC-AGI-2 | v11 (worker hash seed) | 29.31 | **30.56** | 1.25 | between thresholds; leans Kaggle-side |

**ARC-AGI-3 reproduces to two decimals.** The 14x identical-code spread
measured on v2 -- 0.01, 0.04, 0.12, 0.12, 0.14, 0.15, 0.18 -- was not the
board. It was `hash(self.game_id)` being re-randomised in every Kaggle process.
Kaggle's ARC-AGI-3 rerun is deterministic given our seed.

This retracts the strongest claim in this file. "ARC-AGI-3 is not a measuring
device at all" is wrong: it is a precise one for seeded code. What survives is
narrower and still worth saying -- *any* single-run comparison by anyone whose
agent seeds from `hash()` or the clock is inside that noise, and the reference
starter agent seeds from `time.time()`.

Two consequences cut opposite ways. Seeded local A/Bs should now transfer to
the board in ordering, which makes local iteration meaningful again. And the
lucky 0.18 draws are gone: a fixed seed is one draw from that 0.01-0.18
distribution, and ours landed at 0.03.

**ARC-AGI-2 is the opposite.** Seeding the workers did not collapse the
spread -- 1.25 points on n=2 against 1.67 on five mis-seeded samples. The
residual is Kaggle-side: bf16 GPU kernels are not bitwise deterministic, and
the DFS is cut by wall-clock, so the same code decodes slightly differently
each run. The ~2-point resolution floor stands for that board.

## RETRACTION: every pre-2026-09-10 ARC-AGI-3 ablation was mis-seeded

Agent seeding used `hash(self.game_id)`. Python randomises string hashing per
process unless `PYTHONHASHSEED` is set, so the `--seeds N` machinery that every
ARC-AGI-3 A/B in this file relied on **was not controlling anything across
processes**. Found by Codex on 2026-09-10 and fixed (sha256-derived game seed,
`PYTHONHASHSEED=0` in the child).

Re-running the same three arms with correct seeding:

| arm | true score, before (mis-seeded) | after (correct) |
|---|---|---|
| exploit | 0.1356 | **0.2629** [0.1573-0.3158] |
| explore / sweep | 0.1812 | **0.2648** [0.1626-0.3169] |

Two things follow, and both are worse than a simple "numbers moved".

**The exploit ablation is void.** It concluded "exploit is 25% worse, revert
it", and that reverted a mechanism on the strength of a broken randomisation.
Correctly seeded, exploit and sweep are indistinguishable -- 0.2629 against
0.2648, with ranges almost entirely overlapping. The revert may still be
defensible on simplicity grounds; it is not defensible on that evidence.

**The absolute level moved too** (0.13-0.18 to 0.26), so the earlier
distribution was not the real one. The local 5-seed spread quoted as
[0.1179, 0.3055] describes an uncontrolled process, not seed variation.

What is *not* affected: the leaderboard variance measurements (Kaggle-side, no
local seeding involved), all ARC-AGI-2 work, the recovered scoring formula, and
the four environment facts below -- those are properties of the engine, not of
any run.

The honest summary: **no ARC-AGI-3 ablation in this project was correctly
randomised until 2026-09-10.** Sections below that predate it should be read as
suggestive only.

## The objects agent: submitted on the best of three seeds

Codex submitted `object_agent.py` on 2026-09-10 citing "all-25 public mean
improved 0.315011 to 0.451665". That comparison is **one seed against one
seed**; the three-seed run in the same log had no control arm. Running the
control:

| seed | explore | objects | |
|---|---|---|---|
| 0 | 0.3150 | 0.4517 | +43% <- the figure cited |
| 1 | 0.1626 | 0.1834 | +13% |
| 2 | 0.3169 | 0.2239 | **-29%** |
| **mean** | **0.2648** | **0.2863** | **+8.1%** |

The cited number is the most favourable of three. Levels are more consistent
(6.3 -> 7.7, objects at least matching on every seed), but level count and
score move in opposite directions on this benchmark, which is established
above.

The sharp point is internal consistency. The same round *rejected* a `warm`
variant with: "three-seed mean gain only 6.1%, with a large seed-0 regression."
Objects is a three-seed mean gain of 8.1% with a large seed-2 regression --
the same evidence profile. **By the round's own stated bar it would have been
rejected**; the third seed with a control simply had not been run when the
decision was made.

The mechanism -- resample visible object centroids rather than permanently
excluding cells that once did nothing -- is plausible and may well be a small
gain. It is not established, and it is not +43%.

## ARC-AGI-3: four facts the starter agent ignores

`python agi3/probe.py --games ls20,vc33,ft09,tr87`

**1. `available_actions` is reported every frame and varies per game.**
Of 24 playable games, 6 accept ACTION6 (click) and nothing else; ls20 accepts
only ACTION1–4. The starter samples uniformly from all seven, so on the
click-only games **86% of every action budget is spent on illegal actions**.

**2. Click games have ~2–11 responsive cells out of 4096.**
The engine's internal `_get_valid_actions()` — explicitly never exposed to
agents — reports 2 clickable cells for vc33, 2 for r11l, 2 for lp85, 11 for
tn36. A uniform random click hits one with probability ≈0.05%. Random clicking
is not a weak strategy here; it is arithmetically hopeless. Only systematic
coverage works. (Used locally as an oracle to *validate* discovery methods; the
agent never reads it.)

**3. Part of the frame is a step counter that changes every single step.**
On vc33 it is row 0; on ls20 it is rows 61–62. Over 250 random steps vc33
visits 51 distinct full-frame states — but exactly **1** once row 0 is masked
out. Any novelty-seeking or frame-delta policy that skips this masking is
chasing a counter, not the game. The row position differs per game, so it has
to be detected online.

**4. Single-step frame deltas do not separate valid from invalid clicks.**
Clicking a known-valid cell and a known-invalid cell on vc33 produce
byte-identical deltas: one cell, in the counter row. "Did the frame change" is
only a usable signal *after* masking.

### The scoring formula, recovered

`arc_agi.scorecard.Scorecard.score` is documented "the average score of the
runs" and implemented `max(run.score for run in self.runs)`. Every score
comparison in this file before this section was reading that max. The
per-*game* scores underneath it are correct; only the aggregation is wrong.
`agi3/run.py` now patches a copy of `play_local.py` to emit them and takes the
mean. The engine computes, per level:

```
level_score = min( (human_baseline_actions / actions_on_that_level) ** 2 * 100, 100 )
game_score  = level-index-weighted mean, capped at
              sum(indices of completed levels) / sum(all indices) * 100
```

Human baselines are available locally -- all 25 environments expose
`baseline_actions` (vc33 level 1 is 7 actions; ls20 level 1 is 22).

Three properties matter, and our earlier metrics saw none of them:

**Actions are counted per level, not per game.** Grinding 3000 actions against
level 2 does not dilute what you earned on level 1.

**The efficiency term is squared.** Ten times the human action count returns 1%
of that level.

**Depth sets a hard ceiling.** Completing only level 1 of an eight-level game
caps that game at 1/36 * 100 = 2.78, however efficiently you did it.

Two games make it concrete -- both cleared exactly one level in 3001 actions:

| game | level 1 baseline | score |
|---|---|---|
| lp85 | 17 actions | **2.7778** (at the cap) |
| vc33 | 7 actions | **0.0000** |

lp85 cleared level 1 inside its human baseline and collected the whole ceiling.
vc33 took ~3000 actions to pass a 7-action level, and the square annihilated it.

### The ablation, on the real metric

`python agi3/run.py --ab --steps 3000 --seeds 2`

| arm | levels | progress | **true score** |
|---|---|---|---|
| starter random | 1.5 | 0.0100 | 0.0489 [0.0044-0.0933] |
| ours, argmax | 3.0 | 0.0147 | 0.1141 [constant] |
| ours, sampling | 4.5 | 0.0252 | **0.2153** [0.1265-0.3042] |

The ordering is finally consistent across all three columns, and the sampling
fix is worth ~1.9x over the policy it replaced.

**This also gives the first genuine calibration point.** The argmax policy is
what we submitted; it scores 0.1141 locally and returned **0.15** on the
leaderboard's hidden game set, under the same formula. So local numbers can be
trusted to about that tolerance -- which makes the sampling agent's 0.2153 worth
roughly 0.28 on the board, against a field median of 0.27.

### Depth beats breadth, arithmetically

The cap is the lever. Our agents clear level 1 of several games. Consider
instead clearing levels 1-4 of a single eight-level game at human efficiency:

    cap = (1+2+3+4) / (1+...+8) * 100 = 27.8 for that game
    aggregate contribution = 27.8 / 24 games = 1.16

That single game would be **5x our entire current score**, and the leader sits
at 7.51. Breadth is worth almost nothing here: a first level is the cheapest
index in the weighting, and we are collecting the cheapest one in each of
several games while paying full exploration cost for each.

That is a concrete, testable redirection: spend the action budget going deep in
the games we can already enter, rather than sweeping every game for its
entrance.

### Negative result: exploiting known controls scores worse

The scoring formula says sweeping after you have found a level's controls is
paid for twice -- in actions, and again in the square. So once a cell is known
to move the masked state, replay it instead of finishing the sweep. Frozen as
`agi3/exploit_agent.py`; the only difference from the shipping policy is
`EXPLORE_RATE` (0.15 vs 1.0).

`python agi3/run.py --ab --steps 3000 --seeds 5`

| arm | levels | true score |
|---|---|---|
| starter random | 2.6 | 0.0771 [0.0045-0.1905] |
| **exploit** (EXPLORE_RATE 0.15) | **7.6** | 0.1356 [0.1159-0.1859] |
| **sweep** (EXPLORE_RATE 1.0, shipping) | 6.0 | **0.1812** [0.1179-0.3055] |

**It clears more and scores less.** 7.6 levels against 6.0, across 11 distinct
games against 8 -- and 25% less score. Exploiting known controls grinds through
levels, spending many actions each; plain sweeping occasionally falls into a
level fast and collects a large share of that game's cap (its best seed is
0.3055 against the exploit arm's 0.1859).

That is the depth-versus-efficiency tension made concrete, and it is worth
saying plainly: **on this benchmark, level count and score can move in opposite
directions.** Any agent tuned against levels cleared -- which is the obvious
thing to tune against, and what the library's own summary line shows you -- is
being optimised away from the score.

At 2 seeds this same comparison showed the exploit arm 34% *ahead*. It was
noise. The measurement needs 5 seeds to separate these arms at all, and the
ranges still overlap; only the means are ordered.

The idea is not necessarily wrong -- the implementation trusts a cell as a
control after it moves the masked state once, so one noisy frame can lock the
agent onto a dead cell. Requiring repeated confirmation is the next attempt.

### Where the 30.14 model actually fails, and what that rules out

`python agi2/analyze_submission.py agi2/out/nvfull/submission.json`
(NVARC run over all 120 public evaluation tasks, 167 scored outputs)

| bucket | count | share |
|---|---|---|
| solved | 41 | 24.6% |
| off-by-a-few (>=99% cells) | 8 | 4.8% |
| near-miss (>=90% cells) | 47 | 28.1% |
| right-shape-only | 34 | 20.4% |
| wrong-shape | 37 | 22.2% |

Median cell accuracy 94.1%, p90 100%. A third of all outputs get nine cells in
ten right and score nothing, so the naive ceiling for perfect repair is 57.5%,
2.34x the current score. That number is a trap, and the next measurement is why.

**The two attempts err in overlapping places.** Across the 83 failed outputs
where both attempts have the right shape:

| | |
|---|---|
| error-set overlap (Jaccard), median / mean | **0.50 / 0.51** |
| cells where both attempts agree and are right | 83.2% |
| cells where both **agree and are wrong** | **7.6%** |
| cells where they disagree | 9.1% |
| failed outputs with *fully disjoint* error sets | **8 / 83** |

So roughly half of every error is common-mode. Only 8 outputs have errors a
vote could in principle resolve completely, and resolving them needs at least
three candidates -- two can only tie. The realistic gain from per-cell voting
is about **+4.8 points absolute** (24.6% -> 29.4%), not 2.34x.

**This rules out the obvious mechanism and names the real one.** The public
pipeline aggregates whole grids: `arc_decoder.score_sum` buckets candidates by
`hashable(g["solution"])`, so two answers differing in one cell are treated as
entirely different and compete. Adding per-cell voting is a genuine gap in that
design and it is worth the few lines -- but it is a 5-point idea, not a
doubling.

The 7.6% both-wrong figure is the thing to attack. Those are cells where the
model made the *same* mistake twice, which means it holds a wrong hypothesis
about the transformation rather than sampling noisily around a right one. More
samples of the same distribution cannot fix a misreading. That needs a
different signal -- something that checks a candidate against the
demonstrations rather than against other candidates.

### The selection stage is not where the win is

`repro_nvarc.py --save-candidates` copies the decode store out of the kernel
once (1723 files, 2.5 MB, median 17 candidates per output). `select_offline.py`
then scores any selector against public ground truth in seconds, at zero quota.
Decoding costs ~8 worker-minutes a task; selection costs nothing. Only the
second half is ours, and now only the second half has to be paid for twice.

Three independent attacks on it, all negative:

| selector | solved / 141 | vs upstream |
|---|---|---|
| **upstream `score_full_probmul_3`** | **42** (29.8%) | -- |
| upstream `score_kgmon` | 41 | -1 |
| our per-cell voting (`score_cellvote`) | 36 | **-6** |
| best of 9 alternative ranking statistics | 41 | -1 |
| **ORACLE -- any candidate in the pool** | **51** (36.2%) | **+9** |

**Per-cell voting is wrong here.** A `MERGE_TOL` sweep makes it unambiguous:
0.0 reproduces the baseline exactly (42), validating the implementation, and
every non-zero tolerance is worse, roughly monotonically, to -12 at 0.20.
Candidates that differ slightly are not noisy samples of one hypothesis; they
are distinct answers the ranking already discriminates. The near-miss
statistics that motivated this were real, but they described the *emitted pair*
and not the candidate pool.

**Shape filtering promotes nothing.** For the 9 outputs whose correct answer is
in the pool but ranked below top-2, the truth sits at rank 3 four times and
never below 7. A shape rule inferred from the demos matches the truth in 6 of
those 9 -- and changes the rank of exactly **0**, because the candidates above
it are shape-correct too. The competition is shape-right and content-wrong.

**No reweighting of the available statistics helps.** Nine variants over
`beam_score` and `score_aug` -- counts, sums, means, medians, minima, products
-- and the best reaches 41. Searching nine rankers on 141 outputs would be a
multiple-comparisons hazard if one had won; none did, which makes the
conclusion stronger rather than weaker.

**So the +9 is not reachable from inside the candidates.** Whatever separates
the truth from three shape-correct rivals is not in `beam_score` or
`score_aug`. Getting it needs either better candidates (the decode stage) or a
signal from outside the pool -- the obvious one being leave-one-demo-out:
re-fit on n-1 demonstrations and rank candidates by whether the resulting model
reproduces the held-out one. That is a real verification signal, it is exactly
what this pipeline lacks, and it costs decode-stage compute.

### Four attacks on selection, one conclusion

Everything below runs on the saved candidate store in seconds, so these are
cheap to reproduce and cheap to extend.

| attempt | solved / 141 | vs upstream |
|---|---|---|
| **upstream `score_full_probmul_3`** | **42** (29.8%) | -- |
| per-cell voting over merged clusters | 36 | **-6** |
| filter by demo-inferred output shape | 42 | 0 |
| best of 9 reweightings of the available statistics | 41 | -1 |
| demote candidates identical to the test input | 42 | 0 |
| **ORACLE -- any candidate in the pool** | **51** (36.2%) | **+9** |

The last one is the point: **a correct answer is already in the pool for nine
outputs that we get wrong, and nothing we can compute promotes it.**

Why each failed, since the failures are more informative than the ranking:

- **Voting** assumed near-identical candidates were noisy samples of one
  hypothesis. They are not; they are distinct answers, and merging destroys the
  ranking that already discriminates them. `MERGE_TOL=0` reproduces the
  baseline exactly, so this is the mechanism failing, not the code.
- **Shape** cannot separate anything: for the nine recoverable outputs, a
  demo-inferred shape rule matches the truth six times and moves zero ranks,
  because every candidate above the truth is shape-correct too.
- **Reweighting** has nothing left to extract. `beam_score` and `score_aug` are
  the only per-candidate signals, and nine functional forms over them all land
  at or below upstream.
- **Identity demotion** looked promising -- in six outputs the top-2 contained a
  verbatim copy of the test input, and only 1 of 120 tasks has any demo where
  output equals input, so those copies are provably wrong. Demoting them
  rescues nothing, because the candidate that moves up is also wrong.

The rivals ranked above the truth differ from it by a **median of 5% of
cells** -- often one or two. The selector is being asked to choose between
grids that are nearly identical and it has no information that distinguishes
them.

**Read the small differences with care.** These are single-run measurements on
141 outputs; the leaderboard's own identical-code spread is 5.6%, and the local
figure carries comparable sampling noise. A one-output difference is not a
result. The -6 is; the zeros are, in the sense that four independent mechanisms
all failing to move the number is itself the finding.

**Where this leaves ARC-AGI-2.** Selection is a dead end -- upstream tuned it
well and the remaining headroom is not addressable with the information the
candidates carry. Any further gain has to come from the decode stage: better or
more candidates, or a verification signal computed outside the pool.
Leave-one-demo-out is the obvious one -- re-fit on n-1 demonstrations and rank
candidates by whether the resulting model reproduces the held-out pair -- and
unlike everything above it costs decode-stage compute, which is the expensive
half.

### REFUTED: the deadline-coverage hypothesis

The section below argued that the submission runs 50 minutes from the 12h
wall, that the score drift is tail tasks being lost to it, and that halving the
augmentation count should therefore *raise* the score. The leaderboard said no:

| submission | config | score |
|---|---|---|
| v4 x5, identical | augment 16 | 30.14, 29.72, 28.47, 28.47, 28.89 |
| **v10** | **augment 8** | **27.64** |
| v11 | augment 16, audited | 29.31 |

Augment-8 landed below every one of five baseline samples -- 2.1 sd below their
mean, with the predicted direction being *up*. One sample cannot prove harm,
but it rules out the predicted gain.

Codex's 1h public diagnostic explains why the local evidence misled. Under a
1h cutoff, augment-8 covers 40 outputs to augment-16's 32 and scores 13/172 to
4/172 -- coverage dominates when the run is truncated. On the 26 outputs both
reached, each got 4 right: equal quality per task. So the diagnostic was
*correct about the truncated regime*, and the full 11.83h run is evidently not
in that regime. The "50-minute margin" arithmetic is wrong somewhere -- the
hidden tasks may be faster, or the margin larger -- and the "monotone decline"
that motivated it broke on the fifth sample (28.89 after 28.47).

What survives: per-task work is fixed and not elastic, which v8 established
directly. What does not: the claim that the run is deadline-bound. Nothing was
being lost to the wall, so halving the work only cost quality.

### Per-task work is fixed, and the submission runs at the wall

`--subset 24` holds the task set fixed and changes only the wall-clock budget.
v8 gave those 24 tasks a 4h budget; v7 had given the same tasks a share of a 5h
budget spread over 120. Roughly 10x the time available per task:

| run | budget | tasks | candidates/output | solved (same 34 outputs) |
|---|---|---|---|---|
| v7 | 5h | 120 | 16 | 6/34 |
| v8 | 4h | **24** | 16 | 6/34 |

Identical. v8 also finished early, spending 2.18h of quota against the 8h its
budget allowed.

**So `global_end_time` is a deadline that truncates the queue, not a budget
that gets allocated per task.** The solver does a fixed amount of work per
puzzle -- one epoch of LoRA over 16 augmentations, then a DFS with its own
internal limits -- and stops. Giving it more time changes nothing. "More
per-task compute" is not a lever that exists in this code; changing the *amount
of work* (epochs, augmentation count, DFS width) is, and each of those
multiplies the per-task cost.

Which matters, because both runs measure the same per-task cost:

    v7:  120 tasks, 5h wall, 4 workers   ->  10.0 min/task
    v8:   24 tasks, 1.09h wall, 4 workers -> 10.9 min/task

and the real submission has to answer 240:

    240 tasks / 4 workers x 11 min = 11.0h   against an 11.83h budget

**A 50-minute margin, about 5%.** The submission is not budget-bound, it is
running at the competition's 12h wall, which is presumably why upstream tuned
it exactly here.

That also supplies a mechanism for the score drift nobody had explained.
Identical-code resubmits returned 30.14, 29.72, 28.47, 28.47 -- monotonically
down, which is what losing a few tail tasks to the deadline looks like, not
what symmetric noise looks like.

**The testable consequence, and it is cheap:** if the tail is being lost, then
*reducing* per-task work -- 8 augmentations instead of 16 -- should raise the
score, because finishing 240 tasks slightly worse beats finishing 225 of them
well. That is a rare thing in this project: a hypothesis with a mechanism, a
predicted direction, and an expected size above the board's ~2-point
resolution.

---

## The seed is the lottery ticket, and local ordering is inverted (2026-09-15)

The REVERSAL section above ended with a prediction: with seeding fixed, "seeded
local A/Bs should now transfer to the board in ordering, which makes local
iteration meaningful again". Today's submission refutes it — and the same table
shows what the board actually responds to.

`experiments/score-push-20260914/round.json` declared the reading in advance:
`board(seed1) >= 0.03` means local ordering does not transfer. Board(seed1) =
**0.21**, the highest ARC-AGI-3 score this project has recorded.

| ref | agent | seed | local, 25 public games x 5000 actions | board |
|---|---|---|---|---|
| 56166407 | stable v4 | 0 (local **best** of 5) | 0.3045 | 0.03 |
| 56179026 | stable v4, byte-identical resubmit | 0 | 0.3045 | 0.03 |
| 56204668 | stable v4 | 2 (local 2nd) | 0.1473 | 0.06 |
| 56223681 | stable v4 | 1 (local **worst** of 5) | 0.1177 | **0.21** |

Local order 0 > 2 > 1. Board order 1 > 2 > 0. Spearman rho = -1.0.

At n=3 a perfect inversion is p = 1/3 on its own, so this is weak evidence of an
*inversion* and strong evidence against the transfer claim it was testing: the
seed local measurement recommended is the seed the board likes least, and the
one it ranked last is the one that scored 7x higher.

Two things this separates that the earlier sections ran together.

**The board is deterministic.** Two byte-identical reruns of the seeded v4 agent
returned 0.03 and 0.03. That half of the REVERSAL stands.

**The seed still moves the score 7x** (0.03 to 0.21). Determinism was being read
as "the board is a measuring device". What it actually means is that the board
is a *deterministic step function of the agent's trajectory*, and the seed
chooses which step you land on: one extra level cleared in one game is the whole
distance between 0.03 and 0.21. The metric has no resolution between those two
points, only discrete jumps. It also revises the REVERSAL's closing claim that
"the lucky 0.18 draws are gone" — the luck did not disappear with the RNG, it
moved into the seed.

**What changes.** Local ARC-AGI-3 evaluation is retired as a ranking device: it
does not order agents, it does not order seeds, and three paired (local, board)
points now disagree in ordering. The public 25 games and the hidden set are
different distributions and no amount of local tuning closes that.

What replaces it is not a better local metric, it is the board slot itself.
ARC-AGI-3 allows 1 submission/day and counts the best 2 across the whole
competition, so the daily slot becomes: draw an **unsampled** seed for the
stable agent, submit it, keep the two highest. Never spend the slot on a draw
already known to be bad, never drop a good one. The returned number is a draw,
not a capability estimate, and no A/B may be run on it at n=1.

---

## The ensemble is dead at the oracle, and 27.64 landed twice (2026-09-15)

Two closures in one round, both cheap, both predeclared.

**The 2B ensemble has a ceiling it cannot be built to reach.** The 2025 winner's
structural difference from the public recipe is an ensemble, which made "add a
second model" the last untried decode-stage lever. `agi2/ensemble_oracle.py`
answers whether it is worth building from artifacts already on disk — the 4B
store (`agi2/out/v8`) and the 2B store (`agi2/out/exp-ep2/result`), 34 outputs
that both scored:

| | correct |
|---|---|
| oracle(4B) — any 4B candidate is right | 9/34 |
| oracle(2B) — any 2B candidate is right | 9/34 |
| **oracle(4B\|2B)** — the ensemble's ceiling | **11/34** |
| selector, 4B alone | 6/34 |
| selector, 2B alone | 6/34 |
| selector, merged pool | 6/34 |
| selector, cross-agree first — the actual mechanism | 6/34 |

The two models are not redundant: each holds 2 correct answers the other does
not. So a ceiling of +2 tasks is real, and every selector we can name collects
**none** of it — merging the pools and preferring grids both models produced
both return exactly what the 4B selector returns alone. The gap between the
oracle line and the selector line is the same selection failure documented in
*Four attacks on selection*, now measured on a pool that was designed to make it
easier. The ensemble dies here, at zero build cost, instead of after a week of
wiring one up under a 12-hour wall.

Two caveats that limit how far this travels. n=34 is small; a +2/34 ceiling is
not distinguishable from zero at this n, and the honest statement is "no
detectable headroom", not "no headroom". And the 2B store was produced by a
4-hour run against the 4B store's longer budget, so more samples could shift the
ceiling — though not the selector behaviour, which is the binding constraint.

**The gate fired again, identically.** The ARC-AGI-2 slot returned 27.64 for
augment8 v10, byte-identical to the first sample (56140934) and against an
augment16 band of 29.31/30.56/30.97. Predeclared: `<= 28.0` kills dynamic-16/8.
It did. Note the shape of the two distributions rather than their centres: three
reruns of the *same* augment16 notebook span 1.66 points, while two reruns of
the same augment8 notebook agree to 0.00. Board-side variance is not a property
of the notebook, and one point is not a measurable difference on this board.

**A protocol trap worth naming.** Kaggle's CLI prints `N submissions remaining
today.` on **success** — it reports the quota left *after* the write. Read as a
refusal, it declares a quota outage that did not happen and hides a submission
that did. This round it cost a duplicate attempt (refused, harmless) and one
wrongly-recorded "quota did not reset" claim. Acceptance must be decided by
reading the submission list back and matching the description; the CLI's text is
not evidence either way. `scripts/fire_draws.py --self-test` encodes the four
outcome combinations that make this concrete.

## The draw policy paid off, and it was a draw (2026-09-16)

Seed 4 of the *unchanged* stable agent (kernel v8, only `ARC_SEED` differs from
seed 3's build) returned **0.26** — the project's best ARC-AGI-3 score, above
seed 1's 0.21 and to within a hair of the field median (0.27 when last read,
2026-09-04). Ref 56268909, COMPLETE in ~2.5 h.

| seed | 0 | 1 | 2 | 3 | **4** |
|---|---|---|---|---|---|
| board | 0.03 | 0.21 | 0.06 | 0.04 | **0.26** |

Read it as the policy working, not the agent improving: 2 of 5 seeds happened to
clear a level in a game the other 3 missed, which is exactly the deterministic
step function described above — the score has no resolution between 0.03 and
0.26, only discrete jumps. The two slots that count are now 0.26 + 0.21.

What this does *not* change: no ARC-AGI-3 ablation is licensed by it (n=1 per
seed, and the local ordering inversion stands), and the rank prizes stay out of
reach (board top ~7.5). It also does not touch the ARC-AGI-2 conclusion that
~2 points is the resolution floor there — the two boards fail in opposite ways.

---

## Draw six, a lost day, and the cost of a dead watcher (2026-09-17 → 2026-09-19)

Both 2026-09-17 slots came back inside their bands, so nothing was selected
differently: the ARC-AGI-2 pair stays **30.97 + 30.56**, the ARC-AGI-3 pair stays
**0.26 + 0.21**.

| slot | ref | score | predeclared bar | outcome |
|---|---|---|---|---|
| ARC-AGI-2 v11 draw #6 | 56292967 | 30.14 | beat 30.97 to change the pair | pair unchanged |
| ARC-AGI-3 seed 5 | 56292969 | 0.16 | beat 0.21 to change the pair | pair unchanged |

The two bands — six draws for ARC-AGI-2 (a seventh is still running), seven for
ARC-AGI-3:

| | draws | range | above the pair's 2nd slot |
|---|---|---|---|
| ARC-AGI-2 v11 | 29.31, 30.56, 30.97, 28.89, 29.31, 30.14 | 2.08 pts | 1 of 6 |
| ARC-AGI-3 stable seeds | 0.03, 0.21, 0.06, 0.04, 0.26, 0.16, 0.06 | 0.23 | 1 of 7 |

Neither is a measurement of the method — both are byte-identical code. Read
together with the 1.4%-vs-12x spread on identical resubmits above, the ARC-AGI-2
band is a re-roll of a run whose own seed changes a fraction of the answers, and
the ARC-AGI-3 band is the step function.

**2026-09-18 was lost**: neither slot was fired, and an unused slot cannot be
banked. It is the only day of the campaign so far with no submission on either
board.

**2026-09-19 fired both**: ARC-AGI-2 v11 draw #7 (ref 56348699, PENDING, 12 h run
so no result can exist before ~15:45Z) and ARC-AGI-3 seed 6 (kernel version 10,
sha256 c3099d23…, ref 56348735). Seed 6 came back **0.06** at 07:28Z, ~2.5 h
after submit: below the 0.21 bar, so the selected pair is unchanged — recording
it and moving on, which is what was predeclared.

**The record-keeping failure is the transferable lesson.** The 9/17 ARC-AGI-2
watcher logged its opening sleep line and then nothing — `wait-56292967.log` is
one line long, and its result file was never written; the number was only
recovered by reading the board back three days later. (The ARC-AGI-3 watcher
beside it did finish and recorded 0.16, so this is a process-death problem, not a
design one.) `scripts/wait_for_score.py` was written precisely to stop this ("the
number is on disk either way"), and it did, for as long as the process lived.
What a lost watcher actually costs is not the answer — the board keeps that — but
the link between a predeclared bar and the number that came back, which is the
only thing that makes a draw evidence rather than trivia. Mitigation used from
this round on: re-read any ref whose result file is missing with `--once`, before
firing the next day's slot.

Leaderboard refreshed the same day (both fields grew, our scores did not):

| competition | teams | our rank | our score | field median | field max |
|---|---|---|---|---|---|
| ARC-AGI-2 | 2085 | 448 | 30.97 | 28.06 | 76.94 |
| ARC-AGI-3 | 3152 | 1729 | 0.26 | 0.29 | 18.81 |

---

## What this gives the writeup

- **Theory**: a stated hypothesis (the ARC-AGI-2 bottleneck is exact
  completion, not coarse structure), a measurement that supports it, and a
  mechanism it recommends.
- **Progress**: two negative results. Depth-3 composition transfers *zero* to
  ARC-AGI-2 eval. And systematic exploration, which is provably right for
  click-only ARC-AGI-3 games, *loses* to uniform random on action-rich ones —
  a trap worth naming, since the three-game subset we first measured on hid it
  completely. The four environment facts are directly reusable by anyone.
- **Completeness**: every number above has a command next to it.
