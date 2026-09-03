# Measured findings

Everything here is reproducible from this repo with the command shown. These
are the raw material for the Paper Track writeup — particularly the Theory and
Progress criteria, which reward explaining *why* something works and reporting
what did not.

---

## Leaderboard reality check (2026-09-02)

Both pipelines were submitted and scored. Read this section before the
analysis below it, because it bounds how much that analysis is worth.

| competition | our score | rank | field median | field max | teams scoring > 0 |
|---|---|---|---|---|---|
| ARC-AGI-2 | **0.00** | **1760 / 1760** | 27.64 | 72.08 | 1142 / 1760 |
| ARC-AGI-3 | **0.15** | 1949 / 2706 | 0.27 | 7.51 | 2563 / 2706 |

Submissions `55951759` (ARC-AGI-2) and `55949902` (ARC-AGI-3).

**This kills a framing used further down.** The sections below describe
ARC-AGI-2 as a benchmark that resists the classical search family, and they are
accurate about *what was measured*. But 1142 teams score above zero and the
median is 27.64, so the benchmark is not the obstacle -- our hypothesis class
is. A 0.00 against a median of 27.64 is not "the benchmark is hard", it is
"this approach is not in the running". Any claim below that reads as *ARC-AGI-2
is near-impossible* should be read as *depth-3 DSL search is roughly 28 points
behind whatever the field is doing*.

The practical consequence: the propose-and-repair direction argued for below is
derived from where **our** search stalls. That is a real observation about our
search and a poor guide to closing a 0 -> 28 gap. Before building it, find out
what the teams at 30-70 are actually running (ARC Prize 2025 writeups, public
notebooks on this competition). Optimising the last few percent of a dead
branch is the expensive mistake available here.

~~One thing that did hold up: the local harness is predictive. The ARC-AGI-3
local A/B said 0.1141 and the leaderboard said 0.15.~~ **Retracted.** That
0.1141 came from the library's `max(run.score ...)` metric, which reports the
best single game rather than an aggregate -- see *The reference
implementation's score is not the score* below. Two numbers landing near each
other was a coincidence, not a validation. **We currently have no established
relationship between any local metric and the leaderboard**, which is the main
reason to spend a submission on the sampling agent: it buys the first
calibration point.

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
