# What the public record says about the 2026 field (read 2026-09-19)

Survey of public repos, Kaggle discussions and third-party trackers, done while
our own score sat at 30.97 (rank 448/2085). Purpose: find out what the teams
ahead of us are actually doing, and which of it is transferable under
notebook-only / offline / 12 h / 4xL4.

Every number below was read from a listed source today; the sources are given
inline because half of them also carry the command that reproduces them.

## The board, and where the cliffs are

`kaggle competitions leaderboard -c arc-prize-2026-arc-agi-2 -d` (2085 teams):

| rank | team | score | subs |
|---|---|---|---|
| 1 | rabbithole | 76.94 | 91 |
| 2 | Tufa Labs | 73.06 | 27 |
| 3 | nvbanana | 72.08 | 119 |
| 4 | Yi-Chia Chen | 55.14 | 11 |
| 5-9 | Kha Vo / Junhua Yang / Nguyen / AI Winter / Hieu Vy | 37.50 → 34.44 | 52-171 |
| 10-15 | Team BlackBox, Mithil A Vakde, iga, ... | 33.89 → 33.47 | 17-131 |

Two things worth naming. **Four teams are above 55 and the fifth is at 37.5** —
the field is not a smooth curve, it is two populations. And **everything at or
below ~34 sits in the 2025 NVARC recipe space**: 33.89 is the number the public
perfpatch notebook itself claims, and 33.61-33.89 is where three separate teams
have converged. Our 30.97 is inside that population but ~3 points below its
ceiling.

`nvbanana` is CPMP + Ivan Sorokin of NVIDIA, i.e. two of the five authors of the
2025 winning NVARC solution. The people who published the recipe we run are 41
points above us with it.

## Repos worth reading, and what each one teaches

### 1ytic/NVARC — the lineage we are already in (updated 2026-09-15)
<https://github.com/1ytic/NVARC> · read: `README.md`, `ARChitects/`, `TRM/`, `SDG/`

The 2025 winning solution, in three published pieces: a synthetic-data pipeline
(103k synthetic puzzles, 3.2M augmented variants — all public as Kaggle
Datasets), an improved ARChitects Qwen3-4B LoRA fine-tune, and an improved Tiny
Recursive Model. The base model of the whole public lineage is
`sorokin/qwen3_4b_grids15_sft139` — **4B parameters**.

Transferable: this is our floor, and its ceiling is ~34. Nothing in it explains
55-77.

### confluence-labs/arc-agi-2 — what the "agentic coding" family can reach
<https://github.com/confluence-labs/arc-agi-2> · 97.92% on the ARC-AGI-2 public eval

Gemini-CLI agents in E2B sandboxes: `GEMINI_CLI_AGENTS=12` agents per test input,
`MAX_ITERATIONS=10` refinement loops, `CONCURRENCY=132`, 12 h wall clock. Not
runnable in our competition (needs internet + an API key), so it is a *ceiling
estimate*, not a recipe; but it is the clearest evidence that program-synthesis
search over a strong model is where 90%+ lives.

### epang080516/arc_agi — LLM-assisted program synthesis, best perf/cost
<https://github.com/epang080516/arc_agi> · 26.0% ARC-AGI-2, 77.1% ARC-AGI-1

DreamCoder-inspired system with an expanding library of learned programs (LPN);
writeup: <https://ctpang.substack.com/p/arc-agi-2-sota-efficient-evolutionary>.
Generation costs API calls, but candidate *verification* is local and exact —
the part that could be swapped for a local model under the offline rule.

### kschweig/transductARC — cheap experiments on the transduction family
<https://github.com/kschweig/transductARC> · ~76% ARC-1, ~17% ARC-2, 4 h on 1xH100

A plain transformer trained from scratch, with a `WRITEUP.md` full of ablations.
Its list of wins is a checklist of cheap tricks: scale augmentation on top of
colour+dihedral, a second demonstration of the same task in context, bidirectional
attention over the context, trading depth for width, multi-token prediction
(which also buys self-speculative decoding), and symbolic guards to filter
candidate solutions. It descends from `mvakde/mdlARC` — and Mithil Vakde is
rank 11 on our board at 33.89.

### Tufalabs/duck-harness + TAAF — the AGI-3 milestone winner
<https://github.com/Tufalabs/duck-harness> · writeup: Kaggle discussion 717133

Qwen3.6-27B-FP8 behind a tool-using harness that evicts old context so a game
can be played indefinitely, on vLLM. Two lines matter for us beyond the method:
their own submission scored 0.77-1.30 on identical code, and Kaggle retracted a
1.30 result. The board-variance problem we measured on ARC-AGI-3 is not ours
alone; it is the board.

## What the discussion threads settle

- **Runtime is not the lever.** A third-party tracker
  (`tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2`, which
  exposes per-submission runtime that Kaggle's API never shows) has the leader's
  median submission at **562 min against a 720 min cap**, with all nine of his
  improvements (49.17 -> 72.08) inside a 515-605 min band, while rank-2 runs
  longer (726) and scores lower. Source: `myclew` knowledge/strategy notes,
  <https://github.com/seshurajup/myclew>.
- **The leaders dropped test-time training.** cm391's forum inference, quoted in
  the same notes: no per-task gradient steps, so the full 12 h goes to search.
  Unconfirmed, but it is the single load-bearing architectural question, because
  our entire pipeline is TTT and it is time-saturated.
- **Model size alone buys nothing.** Kaggle discussion 732723 runs two 35B MoE
  post-trains (`nex-agi/Nex-N2-mini`, `XYZAILab/XYZ-Aquila-mini`; Qwen3.5-MoE,
  ~3B active) offline on 4xL4 through an identical harness: **0 of 64 evaluation
  tasks solved**, though 96.9% wrote *an* answer and 65.6% got the grid size
  right. The harness (two exemplars, a size hint, and forcing an answer block
  after the reasoning — without it, every reply dies inside `<think>` at the
  token cap) is what converts a model into a score. Same thread: W4A16 int4 on
  the experts only shrinks a 35B MoE from 65.4 GiB to 22.8 GiB at equal score,
  which is how a model of that class fits in 96 GB at all.
- **Make the pipeline deterministic before tuning it.** Kaggle discussion 729743
  publishes a batch-invariant rebuild of the NVARC notebook
  (`thinking-machines-lab/batch_invariant_ops`), precisely so that a CV is an
  experiment and not a reroll, plus per-puzzle adapter caching so a second run
  skips the first run's fine-tuning. This is the answer to the problem we have
  been working around with best-of-draws.
- **The daily cap is real, and the 5/day window was a platform error.** Kaggle
  discussion 705405 (2026-06-09): the competition was mistakenly set to 5/day for
  13 days (2026-05-27 to 06-08) and the surplus submissions were invalidated
  afterwards. `competition_get_submission_limits` agrees with the rules today:
  `numToday: 1`, `numAllowedNow: 0` on both boards. Any team with >1 submission
  per active day accumulated that rate inside that window, not since.

## What this means for us, ranked

1. **Close the ~3-point gap inside our own family first.** The public perfpatch
   lineage claims 33.89; our fork of it draws 27.6-31.0. Before buying into a new
   architecture, one slot on the unmodified configuration answers whether the gap
   is our augmentation, our reduced per-task budget, or the other team's luck.
   This is the cheapest test on the board.
2. **Make the run batch-invariant.** Until it is, every slot is partly a coin
   flip and no CV is attributable — our own findings say a 1-point change is not
   measurable. This costs no submission slot, only an engineering pass.
3. **Then the model class, not the knobs.** If the 55+ population is "strong
   model + search instead of TTT", no amount of tuning our 4B TTT pipeline
   reaches it; the move is a 27-35B open-weight model in int4 behind a harness
   that forces an answer, with the freed budget going to search. That is a
   research project measured in weeks, and the 70-day window is the real budget.
4. **The writeup pools stay the reachable money** (Paper Track $450k, ARC-AGI-2
   Grand Prize $275k, both rubric-scored). This survey is material for the
   Novelty/Completeness criteria — a measured account of the field's cliffs,
   including the two negative results on our own levers, is exactly what a
   writeup pool rewards and a rank pool does not.
