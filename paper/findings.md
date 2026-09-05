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

### ARC-AGI-3: our local metric anti-predicted the board

This is the more instructive result.

| version | local true score (5 seeds) | leaderboard |
|---|---|---|
| v1, argmax policy | 0.1141 | **0.15** |
| v2, sampling policy | 0.1812 | **0.04** |

Local said the sampling policy was 59% better. The board says it is 73% worse.
The metric we spent a day building -- recovering the real formula, caching
human baselines, fixing the library's max-vs-mean bug -- **inverted** against
the only ground truth that counts.

The likeliest explanation is not that the metric is wrong but that **one run
cannot resolve this difference**. Our own 5-seed spread for a single policy was
[0.1179, 0.3055], a 2.6x range. The leaderboard is a *single* run of that same
high-variance measurement, and 0.04 vs 0.15 sits comfortably inside it. Two
policies whose 5-seed means differ by 1.5x are simply not separable by one
sample each.

**The operational consequence is hard.** ARC-AGI-3 allows one submission per
day. If single-run variance is ~2.6x, then no change smaller than roughly 3x
can be confirmed on the leaderboard at all, and a day of submissions buys one
sample. Local iteration cannot be validated against the board at the effect
sizes we are producing.

Two ways out, and they are the real choice on this track:

1. **Stop chasing small gains.** Only pursue changes with a plausible >3x
   effect -- which, given the cap structure, means going deep in a game rather
   than entering more of them.
2. **Treat the local 5-seed mean as the target and stop spending submissions to
   check it.** Accept that we are optimising a proxy, and only submit when the
   local number has moved by more than its own spread.

Neither is comfortable, and pretending the 0.04 was informative about the
policy would be worse.

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
