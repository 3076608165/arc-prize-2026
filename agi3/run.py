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
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "ARC-AGI-3-Kaggle-Starter"
TARGET = STARTER / "agent" / "my_agent.py"
AGENTS = {
    "explore": ROOT / "agi3" / "my_agent.py",
    "random": ROOT / "agi3" / "random_agent.py",
    # Frozen previous policy, kept so a change can be ablated against the thing
    # it replaced rather than only against the random floor.
    "argmax": ROOT / "agi3" / "argmax_agent.py",
}
AB_ARMS = ("random", "argmax", "explore")

SUMMARY = re.compile(r"^\s+(\w+)\s+levels=\s*(\d+)\s+actions=\s*(\d+)\s+state=(\S+)")
SCORE = re.compile(r"Aggregate scorecard score:\s*([\d.]+)")

#: Levels needed to beat each game, cached from the engine (183 across 25 games).
WIN_LEVELS = json.loads((ROOT / "agi3" / "win_levels.json").read_text(encoding="utf-8"))


def progress(per_game: dict) -> float:
    """Mean fraction of each game completed -- our own metric, and the honest one.

    The library's `Scorecard.score` is documented as "the average score of the
    runs" and implemented as `max(run.score for run in self.runs)`. It reports
    the single best game, so it barely moves when a change clears twice as many
    games, and one lucky game swings it by 3x between otherwise identical runs.
    Every score comparison in this repo before this function was reading that.

    This is what the competition actually describes: per-game progress,
    averaged across games. It ignores the action-efficiency term (we have no
    human action counts), so treat it as an upper bound on the real score --
    but it is stable, interpretable, and moves when the agent improves.
    """
    if not per_game:
        return 0.0
    fracs = [
        min(g["levels"] / WIN_LEVELS[gid], 1.0)
        for gid, g in per_game.items()
        if gid in WIN_LEVELS
    ]
    return sum(fracs) / len(fracs) if fracs else 0.0


def venv_python() -> Path:
    for cand in (STARTER / ".venv" / "Scripts" / "python.exe", STARTER / ".venv" / "bin" / "python"):
        if cand.exists():
            return cand
    raise SystemExit(f"No venv under {STARTER}. Run: bash scripts/setup.sh")


def play(agent: str, games: str, steps: int, seed: int = 0) -> dict:
    src = AGENTS[agent]
    if not src.exists():
        raise SystemExit(f"missing agent source {src}")
    shutil.copyfile(src, TARGET)

    cmd = [str(venv_python()), "scripts/play_local.py", "--max-steps", str(steps)]
    if games:
        cmd += ["--game", games]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ARC_SEED": str(seed)}
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
    print(f"\n--- {label} ---  progress {progress(res['games']):.4f}"
          f"   levels cleared {total}"
          f"   lib-score(max-of-games) {res['score']:.4f}")
    for gid, g in sorted(res["games"].items()):
        flag = "  <-- progress" if g["levels"] else ""
        print(f"    {gid:6} levels={g['levels']:2}  actions={g['actions']:5}  {g['state']}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="explore", choices=list(AGENTS))
    ap.add_argument("--games", default="", help="comma-separated ids; empty = all 24")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ab", action="store_true", help="run random then explore")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat the whole A/B under N randomisations and report "
                         "the spread. One run is a noisy sample.")
    args = ap.parse_args()

    def levels(res: dict) -> int:
        return sum(g["levels"] for g in res["games"].values())

    if args.ab:
        rows: dict[str, list[tuple[int, float]]] = {a: [] for a in AB_ARMS}
        cleared: dict[str, set[str]] = {a: set() for a in AB_ARMS}
        for seed in range(args.seeds):
            for arm in AB_ARMS:
                res = play(arm, args.games, args.steps, seed=seed)
                rows[arm].append((levels(res), progress(res["games"]), res["score"]))
                cleared[arm] |= {g for g, v in res["games"].items() if v["levels"]}
                if args.seeds == 1:
                    report(f"{arm} ({args.steps} steps/game)", res)

        if args.seeds > 1:
            print(f"\n=== {args.seeds} seeds x {args.steps} steps/game ===")
            for arm in AB_ARMS:
                lv = [r[0] for r in rows[arm]]
                pr = [r[1] for r in rows[arm]]
                sc = [r[2] for r in rows[arm]]
                print(f"  {arm:<8} levels {lv} mean {sum(lv)/len(lv):.1f}"
                      f" | progress mean {sum(pr)/len(pr):.4f}"
                      f" [{min(pr):.4f}-{max(pr):.4f}]"
                      f" | lib-score(max-of-games) mean {sum(sc)/len(sc):.4f}"
                      f" [{min(sc):.4f}-{max(sc):.4f}]")
                print(f"           ever cleared: {', '.join(sorted(cleared[arm])) or 'none'}")
        else:
            print()
            for arm in AB_ARMS:
                lv, pr, sc = rows[arm][0]
                print(f"  {arm:<8} levels {lv:2}  progress {pr:.4f}  lib-score {sc:.4f}")
    else:
        report(f"{args.agent} ({args.steps} steps/game)",
               play(args.agent, args.games, args.steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
