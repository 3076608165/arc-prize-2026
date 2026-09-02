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

## What we already have to write from

[findings.md](findings.md) holds every measured result with the command that
reproduces it. Two of them are writeup-grade on their own:

- **ARC-AGI-2**: depth-3 compositional search lifts training 3.07% -> 4.83% and
  evaluation 0.00% -> 0.00%. It reaches the right shape on 77% of eval tasks
  and a median 83% of cells (p90 96%), and solves none. The bottleneck is exact
  completion, not coarse structure -- which argues for a propose-and-repair
  method over a wider global search. That is a hypothesis, a measurement, and a
  mechanism: exactly the shape the Theory criterion asks for.
- **ARC-AGI-3**: four environment facts the official starter ignores, and an
  agent built on them that scores 0.1141 against the starter's 0.0505 over all
  24 games -- while clearing one *fewer* level, on a disjoint set of games. The
  split is the interesting part: systematic exploration wins the click-driven
  games and loses the action-rich ones. The facts themselves (legal-action
  masking, ~2-11 responsive cells per click game, the step-counter row, the
  uselessness of raw frame deltas) are reusable by anyone and belong in the
  Progress section, as does the measurement trap -- the three-game subset we
  first tested on hid the regression completely.

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
