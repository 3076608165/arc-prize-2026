# Paper Track writeup — working outline

Hard constraints from the competition page:

- **≤ 1500 words.** Over the limit "may be subject to penalty."
- Kaggle Writeup with title + subtitle, **a cover image is required**
- A **public** notebook attached (a private one auto-publishes after the deadline)
- Optional: a public PDF via Project Link instead of the writeup body
- You must select a Track, and **click Submit** — drafts are not judged
- You will be asked for your **submission ID** to match the notebook to a
  leaderboard entry

Deadline **2026-11-09 23:59 UTC**.

## The rubric is the outline

Six criteria, scored 0–5, **averaged with equal weight**. That last part is the
whole strategy: Accuracy is one sixth, not the headline. A strong leaderboard
score with a thin writeup lands around 2.5–3. The $375k bonus needs > 4.5,
which means five of the six have to be near-perfect.

| Criterion | What the judge is asking | Where it lives |
|---|---|---|
| **Accuracy** | leaderboard score | §Results |
| **Universality** | does this generalize past ARC? | §Discussion |
| **Progress** | does this raise anyone's odds of a top score? | §Discussion, §Negative results |
| **Theory** | *why* does it work, not how | §Method, §Analysis — **the usual failure point** |
| **Completeness** | does the paper cover the actual submission | throughout |
| **Novelty** | new relative to public research | §Related work |

Theory and Novelty are where nearly every competition writeup loses points.
"We fine-tuned X on Y and got Z" is a 2 on Theory no matter how good Z is.

## Structure (~1500 words)

**Title + subtitle** — name the mechanism, not the result.

**Cover image** — one figure that shows the mechanism. Generate it with
`agi2/arclib/viz.py` (`task_to_html` renders demos, truth and both attempts
side by side, which reads well as a figure).

**1. The claim** (~120 w)
One paragraph: what limits current approaches on this benchmark, what we
changed, what that bought. State the leaderboard score and the submission ID
here.

**2. Method** (~450 w)
Enough that someone could reimplement it. Name the representation, the search
or inference procedure, the compute budget, and what runs inside the 12 h /
no-internet envelope.

**3. Why it works** (~350 w) ← *the Theory score*
Not a description of the pipeline. An argument. Ideally:
- a stated hypothesis about *what makes ARC tasks hard* for the prior approach
- a mechanism that addresses that specific thing
- evidence the mechanism is what moved the number — an ablation, a
  stratification by task type, a controlled comparison. One ablation table is
  worth more here than a page of prose.

**4. Results** (~250 w)
Public eval and leaderboard, both. Report the split you tuned on separately
from the one you did not. Include the baseline control — the 3.1% / 0.0% rule
portfolio in this repo is a legitimate floor to cite, and citing an honest
floor reads as rigour, not weakness.

**5. What didn't work** (~150 w)
Cheap points on Progress and it is the section most writeups skip. Negative
results are what let the next person not waste a month.

**6. Related work + what's next** (~180 w)
Position against the public ARC literature — the 2024 and 2025 winning
solutions, test-time training, program synthesis / DSL search, LLM-guided
induction. Novelty is scored *relative to existing public research*, so you
have to show you know what that is.

## What we actually have, and what the paper should be

The score is not the story. ARC-AGI-2 sits at 30.14 (rank 591/1802, above the
field median) and every point of it came from forking a public notebook --
zero on Novelty by construction. Five separate attempts to improve it all
failed. So a paper organised around "our method" has nothing to say.

What this project does have is a run of results about **what can and cannot be
measured on these benchmarks**, each one arrived at by trusting the obvious
measurement and being wrong. That is a real contribution, it is reusable by
everyone in the competition, and it is the rare kind of material that scores on
Theory, Progress and Novelty at once.

**The candidate title: what you can and cannot measure on ARC Prize 2026.**

The evidence, all in [findings.md](findings.md) with reproducing commands:

1. **Leaderboard precision, measured by resubmitting byte-identical code.**
   ARC-AGI-2 returns 30.14 / 29.72 / 28.47 / 28.47 -- a 1.67-point spread, so
   ~2 points is the floor for a detectable change. ARC-AGI-3 returns 0.15 /
   0.04 / 0.12 / 0.01 / 0.14 -- **14x on identical code**. No ARC-AGI-3 change
   below an order of magnitude is confirmable, which means every published
   single-run comparison on that track, ours included, is inside the noise.
   Nobody appears to have measured this.

2. **The reference implementation's own score function is wrong.**
   `arc_agi.scorecard.Scorecard.score` is documented "the average score of the
   runs" and returns `max(...)`. It reports the best single game. Reading it
   inverted one of our ablations: a policy that measured 11x *better* than
   random was in fact slightly worse. Anyone tuning against that number is
   tuning against the wrong thing.

3. **The ARC-AGI-3 scoring formula, recovered and stated.** Per level,
   `(human_baseline_actions / actions_taken)^2 * 100`, capped, then weighted by
   level index and capped again at the completed-level share. Two consequences
   nobody states: level count and score can move in *opposite* directions, and
   completing level 1 of an eight-level game caps that game at 2.78 however
   efficiently you do it.

4. **Selection on ARC-AGI-2 is closed, with an oracle bound.** Five mechanisms
   -- per-cell voting, shape filtering, nine reweightings, identity demotion,
   augmentation-stratified agreement -- all land at or below upstream's
   42/141, against an oracle of 51/141. The correct answer is already in the
   pool for nine outputs and nothing computable promotes it: the rivals ranked
   above it differ by a median of 5% of cells and are shape-correct.

5. **The pipeline runs at the competition's wall, not inside it.** Per-task
   work is fixed, not elastic -- 10x the wall-clock per task produces identical
   candidates and identical accuracy. At 11 min/task, 240 tasks on 4 workers is
   11.0h against an 11.83h budget: a 5% margin. That supplies the mechanism for
   the score drift, which is monotone rather than symmetric.

**What the writeup still needs.** A mechanism of our own that survives its own
ablation. The augmentation-count trade (item 5) is the first candidate with a
mechanism, a predicted direction and an expected size above the resolution
floor. If it works it is the Theory section; if it fails it joins the negative
results, which are already the strongest part of the paper.

## Reference material

- **[state-of-the-art.md](state-of-the-art.md)** — the 2025 podium, the current
  field, and the exact public recipe at LB 33.89, with hyperparameters. Start
  here: Novelty is scored relative to existing public research, so the writeup
  has to show it knows what that is.
- ARC Prize 2025 solution writeups on Kaggle
- 2024 winning solutions on arcprize.org
- <https://arcprize.org/guide>

## Practical notes

- The public notebook is judged as part of Completeness. Make it readable —
  `agi2/build_notebook.py` produces a single inlined cell, which is right for
  *submitting* but poor for *reading*. Consider publishing a second, annotated
  notebook as the paper's artifact.
- ARC-AGI-2 prizes require open-sourcing anyway, so there is no tension between
  competing and publishing.
- Ties break toward whoever **entered first**. If the writeup is done, submit
  it; you can keep editing after.
