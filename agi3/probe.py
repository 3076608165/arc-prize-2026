"""Drive the ARC-AGI-3 env directly and report what the agent's signals look like.

`play_local.py` only prints levels and a score. When both are zero for every
policy you try, you cannot tell whether the exploration is weak or whether the
game is simply not responding to anything you did. This measures the layer
underneath: how often each action changes the frame, how many distinct states
we reach, and -- for click games -- whether object centroids do better than
random pixels.

    python agi3/probe.py --games ls20,vc33 --steps 400
"""
from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "ARC-AGI-3-Kaggle-Starter"

PROBE = r'''
import logging, random, sys, json
from collections import defaultdict, Counter
logging.disable(logging.INFO)
sys.path.insert(0, "."); sys.path.insert(0, "vendor/ARC-AGI-3-Agents")
import arc_agi
from arc_agi import OperationMode
from arcengine import GameAction, GameState

sys.path.insert(0, r"{agi3}")
from my_agent import _components, _layer, BY_VALUE

GAMES = {games}
STEPS = {steps}

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
report = {{}}
for gid in GAMES:
    env = arc.make(gid)
    if env is None:
        continue
    raw = env.step(GameAction.RESET)
    legal = [BY_VALUE[v] for v in raw.available_actions]
    simple = [a for a in legal if not a.is_complex()]
    clicky = [a for a in legal if a.is_complex()]

    def key(r):
        f = [arr.tolist() for arr in r.frame]
        return hash(tuple(tuple(row) for row in _layer(f))), f

    k, frame = key(raw)
    states = {{k}}
    eff = defaultdict(lambda: [0, 0])
    rng = random.Random(7)
    click_hits = click_tries = 0
    rand_hits = rand_tries = 0
    ncomp = []
    levels = 0
    for i in range(STEPS):
        comps = _components(_layer(frame)) if clicky else []
        ncomp.append(len(comps))
        # alternate: half the click budget on centroids, half on random pixels,
        # so the two are measured under identical conditions
        if clicky and (not simple or i % 2 == 0):
            a = clicky[0]
            if comps and i % 4 == 0:
                x, y, _, _ = comps[rng.randrange(len(comps))]
                mode = "centroid"
            else:
                x, y = rng.randrange(64), rng.randrange(64)
                mode = "random"
            a.set_data({{"x": int(x), "y": int(y)}})
        else:
            a = rng.choice(simple or clicky)
            mode = "simple"
            if a.is_complex():
                a.set_data({{"x": rng.randrange(64), "y": rng.randrange(64)}})
        r = env.step(a)
        nk, frame = key(r)
        changed = nk != k
        eff[a.name][1] += 1
        eff[a.name][0] += changed
        if mode == "centroid":
            click_tries += 1; click_hits += changed
        elif mode == "random":
            rand_tries += 1; rand_hits += changed
        states.add(nk)
        k = nk
        levels = max(levels, r.levels_completed)
        if r.state is GameState.GAME_OVER:
            r = env.step(GameAction.RESET); k, frame = key(r)

    report[gid] = {{
        "legal": [a.name for a in legal],
        "distinct_states": len(states),
        "levels": levels,
        "effect": {{n: f"{{v[0]}}/{{v[1]}}" for n, v in sorted(eff.items())}},
        "avg_components": round(sum(ncomp) / max(len(ncomp), 1), 1),
        "centroid_click_change_rate": f"{{click_hits}}/{{click_tries}}",
        "random_click_change_rate": f"{{rand_hits}}/{{rand_tries}}",
    }}

print("@@@" + json.dumps(report))
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="ls20,vc33,ft09,tr87")
    ap.add_argument("--steps", type=int, default=400)
    args = ap.parse_args()

    py = STARTER / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        py = STARTER / ".venv" / "bin" / "python"

    src = PROBE.format(
        games=repr([g.strip() for g in args.games.split(",") if g.strip()]),
        steps=args.steps,
        agi3=str(ROOT / "agi3"),
    )
    proc = subprocess.run(
        [str(py), "-c", src], cwd=STARTER, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        env={"PYTHONIOENCODING": "utf-8", **dict(__import__("os").environ)},
    )
    line = next((l for l in proc.stdout.splitlines() if l.startswith("@@@")), None)
    if not line:
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        return 1

    import json

    for gid, r in json.loads(line[3:]).items():
        print(f"\n=== {gid}  legal={r['legal']} ===")
        print(f"  distinct states reached : {r['distinct_states']} / {args.steps} steps")
        print(f"  levels completed        : {r['levels']}")
        print(f"  frame changed per action: {r['effect']}")
        if r["avg_components"]:
            print(f"  avg click targets found : {r['avg_components']}")
            print(f"  centroid clicks changed : {r['centroid_click_change_rate']}")
            print(f"  random  clicks changed  : {r['random_click_change_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
