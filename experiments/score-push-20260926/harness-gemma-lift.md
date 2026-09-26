# The agent-class arm: lifting a public ARC-AGI-3 harness (2026-09-26)

Why this exists: the seed lottery and every decode-stage knob are measured null
(13 board draws, 2 above the pair's second member; the ARC-AGI-2 arm family is out of
unmeasured knobs). The one lever nobody in this project has built is an **agent
class** — the board's tens come from LLM-in-the-loop harnesses, ours is a sweep
policy at 0.27. So this arm is a lift, not an invention: the reference says to read a
published harness and lift it, and `references/kernel-experiment-runs.md` documents
the lift pattern.

## What was lifted

`bernicerobinson/arc-agi-3-m2-gemma-reflection` — pulled 2026-09-26 to
`experiments/intel-20260926/arc-agi-3-m2-gemma-reflection/`.

| property | value |
|---|---|
| internet | `False` (offline — legal under the comp's offline rule) |
| accelerator | `enable_gpu=true`, `machine_shape=NvidiaRtxPro6000`, pinned private-BYOD docker image |
| model | `google/gemma-4/Transformers/gemma-4-31b-it/1` (public, free to attach) |
| datasets | `ko0kip/vllm-0230-offline` (4.2 GB of vLLM 0.23.0 wheels) |
| kernels | `philipvonderlind/vllm-deps` |
| agent | 71,564 chars / ~1,680 lines: vLLM-served LLM in the loop, reflection memory, global deadline, action-JSON repair, ineffective-action blacklisting, per-step LLM trace |
| plumbing | the competition's own `ARC-AGI-3-Agents` harness copied from the input mount, `agents/__init__.py` registering `myagent`, `.env` switching offline ↔ `gateway:8001` rerun mode |
| rules check | "the use of external data and models is acceptable unless specifically prohibited by the Host" + winner must open-source system/model/weights; a public Kaggle-hosted open-weight model satisfies both. Score claim of the source notebook: **unverified** — nothing in its title or metadata states one, and this file does not invent one. |

## How it is built (reproducible, not a copy-paste)

`agi3/lift_harness.py` extracts the agent cell to the tracked
`agi3/harness_gemma_my_agent.py` and generates `agi3/out/harness-gemma/` from it,
refusing if the pulled cell and our tracked copy diverge. Our only change to the
notebook is the kernel identity (`id`, `title`, `is_private`); everything else —
sources, docker image, accelerator — is carried verbatim, because those are where a
real configuration difference hides.

```
python agi3/lift_harness.py --self-test   # source cell #7, 11 code cells
python agi3/lift_harness.py               # build
```
Build record: `agi3/out/harness-gemma/experiment.json` (source notebook sha256
`dee15082…`, lifted agent sha256 `da1f1a4a…`, built notebook sha256 `d63d91db…`).

Pushed 2026-09-26 as kernel version **17** of `uts3125025522741/arc-agi-3-sweep-agent`
(no submission slot consumed by a push).

## What the commit run can and cannot prove

Commit mode skips the local 25-game self-test (the notebook's own default), so the
green it can produce is: the wheels dataset and deps kernel resolve, `vllm` and
`arc_agi` import, the official agents harness is in place, and a dummy
`submission.parquet` is written. It does **not** exercise the agent loop — the first
scored rerun is the first measurement of the agent. That is the same shape as every
other arm here: the environment gate is cheap, the behaviour gate is the slot.

## Predeclared reading for the AGI-3 slot that carries it

- `> 0.27` — the class works: new pair member, and the campaign's method for this
  board changes (build on it, one knob per day).
- `0.10 – 0.27` — above the seed median (0.07) but below the pair: the class is
  alive but under-tuned; the next arms are its knobs (prompt, model size, eviction).
- `≤ 0.10` — indistinguishable from the seed lottery; the harness as lifted is not
  the lever, and the failure mode to investigate first is the deadline/eviction path
  (its own logs carry per-step LLM traces).

Either way the slot buys more than another seed would: a seed can only re-draw a
distribution already measured at 2/13, while this is the first reading from a
different distribution.

## How it gets fired (both boards, one slot each per UTC day)

`scripts/fire_next_arm.py` advances the queue by exactly ONE arm per invocation —
correct for one board, insufficient now that two boards each have a daily slot. So
the same idempotent invocation is scheduled twice, 20 minutes apart, and the queue
order does the allocation:

| when (local) | route | fires |
|---|---|---|
| 10:05 daily | Hermes cron `arc-fire-next-arm` + Windows task `ARC fire next arm` | queue head: `twopass-largest-first` (AGI-2, kernel v18) |
| 10:25 daily | Hermes cron `arc-fire-next-arm-b` + Windows task `ARC fire next arm B` | next un-fired arm: `harness-gemma` (AGI-3, kernel v18) |

Both routes read the submission list to decide what is already fired, so the second
invocation skips whatever the first landed, and a day when the first fails degrades to
"the harness waits for tomorrow" rather than to a double spend. Verified 2026-09-26:
both tasks registered with next runs 9/27 10:05 and 10:25, and the dry run under the
cron runner's own interpreter selects `twopass-largest-first` v18.

