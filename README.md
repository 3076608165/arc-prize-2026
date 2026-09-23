# ARC Prize 2026

Working repo for all three 2026 ARC competitions. One local harness for
ARC-AGI-2, the official starter wired up for ARC-AGI-3, and a writeup skeleton
for the Paper Track that both feed.

## Where we actually stand (2026-09-23)

| competition | our score | rank | field median | field max |
|---|---|---|---|---|
| ARC-AGI-2 | **30.97** | 457 / 2156 | 28.06 | 83.06 |
| ARC-AGI-3 | 0.27 | 1751 / 3258 | 0.30 | 18.81 |

Both scores are the best single submission so far (finals take the best two),
read back from the leaderboard files with `kaggle competitions leaderboard -c
<comp> -d`; the rank is the board's own, which is the last position of a tie
group. Both fields grew faster than we scored, so both ranks slipped even though
nothing changed on our side. ARC-AGI-3's 0.27 is a lucky draw from a
deterministic step function, not a capability estimate — see
[paper/findings.md](paper/findings.md).

ARC-AGI-2 went 0.00 -> 30.97 by abandoning our own approach and running the
public NVARC-lineage notebook. See [paper/state-of-the-art.md](paper/state-of-the-art.md).

ARC-AGI-3 is stuck, and [paper/findings.md](paper/findings.md) explains why it
is hard to unstick: our local metric said the newer policy was 59% better and
the board scored it 73% worse, which single-run variance alone can account for.
At one submission per day, changes below ~3x are not measurable there.

### Which board can measure anything

Resubmitting byte-identical code: ARC-AGI-2 returned 30.14 then 29.72 (1.4%
spread); ARC-AGI-3 returned 0.15, 0.04, 0.12, 0.01 (12x). AGI-2 resolves a
one-point change; AGI-3 cannot resolve an order of magnitude. Effort goes to
AGI-2.

## Deadlines (all 23:59 UTC)

| Date | What |
|---|---|
| **2026-10-26** | ARC-AGI-2 / -3 **entry + team merger deadline**. Miss it and you cannot compete at all. |
| 2026-11-02 | ARC-AGI-2 / -3 final submission |
| 2026-11-09 | **Paper Track** final submission |
| 2026-12-04 | Winners announced |

The Paper Track is not standalone: it scores a writeup *about* your ARC-AGI-2 or
ARC-AGI-3 submission, and asks for the submission ID. No leaderboard entry, no
paper entry.

## The one thing to do first

Accept the rules on all three competition pages. It is free, it is reversible,
and it is the only hard gate:

- <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules>
- <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules>
- <https://www.kaggle.com/competitions/arc-prize-2026-paper-track/rules>

## Layout

```
agi2/                 ARC-AGI-2: local harness (pure stdlib, no deps)
  arclib/
    loader.py         task JSON -> Task/Pair
    grids.py          D4 group, bbox, crop, tile, upscale, ...
    dsl.py            parameterised primitives for the search
    search.py         depth-3 BFS + symmetry repair + neighbourhood rule
    solvers.py        rule portfolio, solve_task(), solve_task_full()
    evaluate.py       the exact Kaggle metric, format validator, writer
    viz.py            grids -> SVG/HTML (also: paper figures)
  run_baseline.py     score a split (--full for the search)
  analyze.py          how close does the search get, and where does it stop
  inspect_tasks.py    render tasks + predictions to a browsable page
  build_notebook.py   inline arclib into one offline Kaggle notebook
agi3/
  my_agent.py         our agent (source of truth; synced into the starter)
  random_agent.py     the starter's baseline, kept for A/B
  run.py              sync + play + A/B
  probe.py            instrument the environment itself
paper/
  outline.md          writeup skeleton against the six rubric criteria
  findings.md         every measured result, with the command that makes it
data/                 cloned datasets and vendored repos (gitignored)
scripts/setup.sh      reproduce everything from a clean checkout
```

## Setup

```bash
bash scripts/setup.sh
```

Clones the datasets, checks the ARC-AGI-2 harness, and builds the Python 3.12
venv the ARC-AGI-3 game engine needs (via `uv`).

## ARC-AGI-2

Pure stdlib, no dependencies, whole eval split scores in under a second.

```bash
python agi2/run_baseline.py --split evaluation            # rules only, <1s
python agi2/run_baseline.py --split evaluation --full     # + search, ~60s
python agi2/analyze.py --split evaluation                 # how close does it get?
python agi2/inspect_tasks.py --split training --limit 12 --predict
python agi2/build_notebook.py --self-test
```

### Where we stand

| pipeline | training (1000) | evaluation (120) |
|---|---|---|
| 9 hand-written global rules | 3.07% | **0.00%** |
| `--full`: + depth-3 DSL search, symmetry repair, 3×3 neighbourhood rule | 4.83% | **0.00%** |

Compositional search lifted training by 57% relative and evaluation by nothing.
`agi2/analyze.py` explains why: on evaluation the search reaches the right grid
shape on 77% of tasks and a median of **83% of cells correct** (p90 96%) — and
solves **0%**. The training distribution is identical except for that last row.

So the gap is not coarse structure, it is exact completion, and ARC pays
nothing for 96%. Widening the global hypothesis space moves tasks from
"nothing" to "almost", which scores zero. See
[paper/findings.md](paper/findings.md) for the full table and what it implies.

The 240 hidden leaderboard tasks come from the same distribution as the 120
public eval tasks, so eval is the honest predictor — not training.

What the teams above us are doing, and which of it travels, is surveyed in
[docs/public-solutions-2026-09.md](docs/public-solutions-2026-09.md).

### Kaggle code-competition constraints

- Notebook only, **no internet**, ≤ 12 h wall clock (CPU or GPU)
- Output must be `submission.json`, every `task_id` present, both
  `attempt_1` and `attempt_2` present for every test input
- External data and pretrained weights are fine if freely and publicly
  available — but they must be attached as a Kaggle Dataset/Model, not
  downloaded at runtime
- L4×4 (96 GB) is available; it burns GPU quota at 2×
- **Prizes require open-sourcing your solution.**
- The data mounts at `/kaggle/input/competitions/<slug>/`, **not**
  `/kaggle/input/<slug>/`. Do not hardcode either — recurse from
  `/kaggle/input`, as the generated driver does.

`build_notebook.py` inlines `arclib` into a single cell so the notebook has no
imports to resolve. Edit the Python locally, rebuild, and never touch the
Kaggle editor.

## ARC-AGI-3

The official starter (`data/ARC-AGI-3-Kaggle-Starter`) is set up and verified
against the real game engine — 25 games, ~230 fps. Our agent lives in
[agi3/my_agent.py](agi3/my_agent.py); `agi3/run.py` syncs it in and A/Bs it
against the starter's random baseline.

| agent | levels cleared | aggregate score |
|---|---|---|
| starter random agent | **4** | 0.0505 |
| legal-only + counter-masked + systematic sweep | 3 | **0.1141** |

*(All 24 games, 3000 steps each.)* We are 2.2x ahead on score and one level
behind on count — and the two agents clear **disjoint** sets of games. We win
the click-driven ones (lp85, vc33) where random is arithmetically hopeless, and
lose four action-rich ones (ar25, cd82, m0r0, sp80) that random happens to
stumble through. Three measured facts justify the masking; the systematic
schedule on top of it is not yet a net win. Details, the cause, and the fix in
[paper/findings.md](paper/findings.md).

```bash
python agi3/run.py --steps 3000                    # our agent, all 24 games
python agi3/run.py --ab --steps 3000 --games vc33  # against the random baseline
python agi3/probe.py --games ls20,vc33             # what the env actually does
```

See [agi3/README.md](agi3/README.md) for the Windows-specific notes (the
starter's `Makefile` assumes a POSIX venv layout, and one upstream script
needed an encoding fix).

Scoring is 0–100% per game, where 100% means beating the game in no more
actions than a human took, so clearing a level at all is what separates a
score from a zero.

## Paper Track

See [paper/outline.md](paper/outline.md). The six rubric criteria are weighted
equally, so the shape of the writeup matters as much as the score behind it.

## Provenance

Datasets are cloned from `github.com/arcprize` under their own licenses
(ARC-AGI-2 data: Apache 2.0). `data/` is gitignored; re-create it with
`scripts/setup.sh`.
