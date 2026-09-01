"""Sync an agent into the starter kit and play it, with an A/B mode.

The starter loads `agent/my_agent.py` by path, so the repo keeps the real
source in `agi3/` and copies it over before each run. That keeps our work out
of the gitignored `data/` tree and makes the baseline comparison one command.

    python agi3/run.py --steps 300 --games ls20,vc33,ft09
    python agi3/run.py --steps 300 --agent random      # the starter's baseline
    python agi3/run.py --steps 300 --ab                # both, same budget
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "ARC-AGI-3-Kaggle-Starter"
TARGET = STARTER / "agent" / "my_agent.py"
AGENTS = {"explore": ROOT / "agi3" / "my_agent.py", "random": ROOT / "agi3" / "random_agent.py"}

SUMMARY = re.compile(r"^\s+(\w+)\s+levels=\s*(\d+)\s+actions=\s*(\d+)\s+state=(\S+)")
SCORE = re.compile(r"Aggregate scorecard score:\s*([\d.]+)")


def venv_python() -> Path:
    for cand in (STARTER / ".venv" / "Scripts" / "python.exe", STARTER / ".venv" / "bin" / "python"):
        if cand.exists():
            return cand
    raise SystemExit(f"No venv under {STARTER}. Run: bash scripts/setup.sh")


def play(agent: str, games: str, steps: int) -> dict:
    src = AGENTS[agent]
    if not src.exists():
        raise SystemExit(f"missing agent source {src}")
    shutil.copyfile(src, TARGET)

    cmd = [str(venv_python()), "scripts/play_local.py", "--max-steps", str(steps)]
    if games:
        cmd += ["--game", games]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # The child prints UTF-8; this box's locale is GBK, so decode explicitly.
    proc = subprocess.run(
        cmd, cwd=STARTER, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )

    per_game, score = {}, None
    for line in proc.stdout.splitlines():
        if m := SUMMARY.match(line):
            per_game[m.group(1)] = {
                "levels": int(m.group(2)),
                "actions": int(m.group(3)),
                "state": m.group(4),
            }
        elif m := SCORE.search(line):
            score = float(m.group(1))
    if score is None:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
        raise SystemExit(f"{agent}: run produced no scorecard")
    return {"score": score, "games": per_game}


def report(label: str, res: dict) -> None:
    total = sum(g["levels"] for g in res["games"].values())
    print(f"\n--- {label} ---  aggregate score {res['score']}   levels cleared {total}")
    for gid, g in sorted(res["games"].items()):
        flag = "  <-- progress" if g["levels"] else ""
        print(f"    {gid:6} levels={g['levels']:2}  actions={g['actions']:5}  {g['state']}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="explore", choices=list(AGENTS))
    ap.add_argument("--games", default="", help="comma-separated ids; empty = all 24")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ab", action="store_true", help="run random then explore")
    args = ap.parse_args()

    if args.ab:
        base = play("random", args.games, args.steps)
        report(f"random baseline ({args.steps} steps/game)", base)
        ours = play("explore", args.games, args.steps)
        report(f"explore agent ({args.steps} steps/game)", ours)
        b = sum(g["levels"] for g in base["games"].values())
        o = sum(g["levels"] for g in ours["games"].values())
        print(f"\nlevels cleared: random {b} -> explore {o}")
        print(f"scorecard:      random {base['score']} -> explore {ours['score']}")
    else:
        report(f"{args.agent} ({args.steps} steps/game)", play(args.agent, args.games, args.steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
