# What the 28-point gap actually is

Written 2026-09-02, after our ARC-AGI-2 submission scored 0.00 at rank
1760/1760. The question was whether the gap is a tuning problem or a family
problem. It is a family problem.

## The field

**ARC Prize 2025 podium**, on the ARC-AGI-2 private set:

| | team | score | approach |
|---|---|---|---|
| 1st | NVARC | 24.03% | synthetic-data ensemble of an Architects-style test-time-trained model plus TRM components |
| 2nd | the ARChitects | 16.53% | 2D-aware masked-diffusion LLM, recursive self-refinement, perspective-based scoring |
| 3rd | MindsAI | 12.64% | test-time fine-tuning, augmentation ensembles, tokenizer dropout, pretraining tricks |

**ARC Prize 2026, today**: max 72.08, median 27.64, 1142 of 1760 teams above
zero. The field has roughly tripled the 2025 winner in nine months.

ARC Prize's own 2025 analysis splits the remaining difficulty in two: the
accuracy gap is "bottlenecked by engineering", the efficiency gap is
"bottlenecked by science and ideas". That is a useful steer — accuracy is
currently bought with engineering, not insight.

Not one of these is a program-synthesis or DSL-search system. Our approach and
theirs do not share a family, which is why our score is 0.00 rather than small.

## The public recipe, exactly

The best-scoring *public* notebook is `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`
at **LB 33.89**. `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2`
(202 votes) is the same pipeline; `albartrose/...-best-score-19-17` is a weaker
tune of it. All of them are the NVARC lineage, and all attach the same weights.

**Base model** — `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`,
a public Kaggle Model, so it attaches offline and satisfies the no-internet
rule. Qwen3-4B already supervised-fine-tuned on ARC grids, 7.3 GB in bf16
across two safetensors shards. Note `vocab.json` is **94 bytes**: the tokenizer
has been cut down to the handful of symbols ARC grids actually use.

**Per-task test-time training.** For every task, LoRA-finetune the base model on
*that task's own demonstration pairs*:

```
r=256, lora_alpha=32, lora_dropout=0.0, use_rslora=True
learning_rate=5e-5, num_train_epochs=1, per_device_train_batch_size=1
max_seq_length=8192, load_in_4bit=False
train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
```

Sixteen augmented views of two to six demo pairs, one epoch, per task. This is
the core of the whole family: the model is re-specialised on each puzzle before
it answers.

**Constrained decoding — `turbo_dfs`.** Not beam search. A depth-first walk of
the token tree scored by cumulative NLL against a score threshold, exploiting
the fact that only ~12 tokens are legal in an ARC grid: normalise over those 12
logits on GPU, keep the tree bookkeeping on CPU. A wall-clock `end_time` is
threaded through the recursion so the search degrades gracefully instead of
timing out.

**Selection by augmentation agreement.** Candidate answers are re-scored under
several augmented views of the same task and the scores aggregated, rather than
trusting a single decode.

## What this means for us

1. **There is no path from our 0.00 to 28 by improving the DSL search.** The
   propose-and-repair idea in [findings.md](findings.md) targets where our
   search stalls. Our search is not on the road the field is driving down.

2. **Reproducing the public notebook gets to roughly rank 200/1760, not to
   prizes.** Best public is 33.89; the leaders are at 72.08. The leaders have
   something worth ~2x the public recipe that is not published.

3. **For the Paper Track this matters more than the score.** Novelty is scored
   *relative to existing public research*, and re-running a 202-vote public
   notebook is a 0 on that criterion no matter what it scores. The paper needs
   a mechanism of our own on top of a working baseline.

4. **What survives from what we built.** The measurement layer, not the
   solvers: `arclib/evaluate.py` implements the exact Kaggle metric with a
   format validator, `analyze.py` stratifies where a method stops, and the
   ARC-AGI-3 harness proved predictive of the hidden set (0.1141 local vs 0.15
   leaderboard). Those are the tools for evaluating *any* method. The DSL
   search becomes what it always honestly was — a control condition to cite.

## Hardware, and where iteration has to happen

| | VRAM | fits |
|---|---|---|
| this laptop (RTX 3070 Ti Laptop) | 8 GB | 4-bit QLoRA only; bf16 weights alone are 7.3 GB |
| Kaggle T4 x2 | 2 x 16 GB | bf16, which is what the public notebook uses |
| Kaggle L4 x4 | 96 GB | comfortable, burns GPU quota at 2x |

`torch` is not installed locally. Local work is possible in 4-bit for
debugging the plumbing, but any number that means anything has to come from a
Kaggle run. Note that "Save & Run All" commits are cheap — only clicking
*Submit to Competition* spends one of the 5 daily submissions.

## Reproduction status (validated 2026-09-02)

The fork runs. A 1-hour validation (`--hours 1`, v2 of
`uts3125025522741/arc-agi-2-nvarc-repro`) completed cleanly and wrote a
format-valid `submission.json`: model load, per-task LoRA test-time training,
`turbo_dfs` decode, and submission write all work end to end.

It processed 4 of 120 tasks -- because `starter.py` hardcodes a four-task
allowlist whenever `KAGGLE_IS_COMPETITION_RERUN` is unset, to keep commits
cheap. So it proved the pipeline runs and said nothing about speed.

### Throughput, measured (v3, `--hours 1 --full-eval`)

Dropping that allowlist and running 1h against all 120 public evaluation tasks
gives the number the full-run decision depends on. Cost: 2.11h of quota.

Scored with our own harness against the public ground truth, by queue position
(the queue is `sorted(data.keys())`, so position is time order):

| queue quartile | shape correct | solved |
|---|---|---|
| tasks 1-30 | **27/46 (59%)** | **4/46** |
| tasks 31-60 | 0/42 | 0/42 |
| tasks 61-90 | 0/39 | 0/39 |
| tasks 91-120 | 0/40 | 0/40 |

The cliff is the answer. Roughly **30 tasks got real treatment in one hour on
4x L4**, and everything after the deadline was emitted as filler -- non-blank,
so a naive "did it produce a grid" check counts it as a prediction, which is
how I first misread this run as full coverage.

That fixes the arithmetic:

- 1h x 4 workers = 240 worker-minutes over ~30 tasks = **~8 worker-minutes per
  task**, and the work per task looks fixed rather than elastic.
- 240 hidden tasks x 8 min / 4 workers = **~8 hours**. The 12h limit fits with
  roughly 50% headroom.
- On tasks that were actually processed: **8.7% of outputs solved, 59% of
  shapes correct**. For comparison our depth-3 DSL search gets 77% of shapes
  and 0% solved. The LLM is worse at coarse geometry and strictly better at
  the thing that scores.

8.7% on 46 outputs is a small, non-random sample (the alphabetically first
tasks of the public split, not the hidden 240), so it does not contradict the
reference's 33.89 -- it is one point, at slightly less per-task budget, on a
different set.

## Suggested order## Suggested order

1. Attach the model, run the public notebook unmodified, confirm we reproduce
   ~19-34. This replaces 0.00 with a real baseline and validates the plumbing.
2. Score it on the public 120-task evaluation split with our own harness, so
   we have a local proxy that does not cost a submission.
3. Only then look for the mechanism worth writing about. The 2025 analysis
   says accuracy is engineering-bound and efficiency is idea-bound — and the
   ARC-AGI-2 bonus prize needs 85%, while the Paper Track bonus needs a 4.5/5
   rubric score. Those two point at different work.

## Sources

- [ARC Prize 2025 Results and Analysis](https://arcprize.org/blog/arc-prize-2025-results-analysis)
- [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2)
- Public notebooks, pulled 2026-09-02: `mikelou1/arc-agi2-lb33-89-minimal-perfpatch`,
  `nihilisticneuralnet/baseline-nvarc-arc-25-winning-solution-for-t4x2`
