# ARC-AGI-3 — local notes

The code lives in the official starter at `data/ARC-AGI-3-Kaggle-Starter`
(cloned by `scripts/setup.sh`). This file records what had to change to make it
work on Windows, and what is verified.

## Verified working

```
25 environments fetched from the ARC-AGI-3 API
ls20, vc33 played 50 steps each at ~230 fps
Aggregate scorecard score: 0.0     <- expected for the random starter agent
```

## Two Windows deviations from the starter's README

**1. Don't use `make setup`.** The `Makefile` hardcodes the POSIX venv layout
(`.venv/bin/python`) and a `python3.12` binary on PATH. Neither exists here.
`scripts/setup.sh` does the same work with `uv` and the `Scripts/` layout.

If you do want the Makefile targets, override the paths:

```bash
make play-local VENV_PY=.venv/Scripts/python.exe
```

**2. `scripts/slim_framework.py` had an encoding bug.** It wrote an em-dash into
`vendor/ARC-AGI-3-Agents/agents/__init__.py` using the platform default
encoding. On a GBK Windows box that produces a file Python then refuses to
import:

```
SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xa1
```

`scripts/setup.sh` patches the write to `encoding="utf-8"`. Worth reporting
upstream.

Also set `PYTHONIOENCODING=utf-8` before `play_local.py` if console output
raises `UnicodeEncodeError`.

## The loop

```bash
cd data/ARC-AGI-3-Kaggle-Starter
PY=.venv/Scripts/python.exe

$PY scripts/play_local.py --list                      # 25 game ids
$PY scripts/play_local.py --game ls20 --max-steps 200 # one game
$PY scripts/play_local.py --max-steps 200             # all of them
```

Edit `agent/my_agent.py` — that is the only file. It defines `MyAgent` with
`is_done(frames, latest_frame)` and `choose_action(frames, latest_frame)`.
Games are cached in `environment_files/` after first download, so the loop is
offline and fast.

## Submitting

```bash
mkdir -p .kaggle && echo "KGAT_..." > .kaggle/access_token   # from kaggle.com/settings
# edit notebooks/kernel-metadata.json -> your Kaggle username
make submit VENV_PY=.venv/Scripts/python.exe
make status VENV_PY=.venv/Scripts/python.exe
```

`make submit` only runs Phase A (Save & Run All). You then open the kernel on
kaggle.com and click **Submit to Competition** for a leaderboard score. That
click spends one of **5 daily submissions**, so get `play-local` scoring
non-zero first.

Accelerator is set in `scripts/build_notebook.py` (`ACCELERATOR = "t4"`;
options `cpu`, `t4`, `p100`, `rtx6000`). RTX 6000 is ARC-AGI-3-exclusive and
burns quota fast — don't use it while iterating.

Runtime cap here is **9 hours**, not 12.

## What to actually build

The random agent scores 0 because it never completes a level. The scoring
formula (100% = win every game in no more actions than a human) means two
separable problems:

1. **Win at all.** Needs exploration that discovers the game's rule from
   frames, not a fixed policy. This is where the score comes from first.
2. **Win efficiently.** Only matters once (1) is solved.

`frames` is the full history, so the agent can diff consecutive frames to learn
what each action does — that mapping is the thing worth modelling, and it is
also the thing worth writing about in the paper.
