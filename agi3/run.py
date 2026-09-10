"""Play tracked agent sources with reproducible A/B experiments.

The local runner imports each selected source directly from `agi3/`.
Agents have separate runner files so experiments do not overwrite each other's
source. PYTHONHASHSEED and ARC_SEED are fixed in each child process.

    python agi3/run.py --steps 300 --games ls20,vc33,ft09
    python agi3/run.py --steps 300 --agent random      # the starter's baseline
    python agi3/run.py --steps 300 --ab                # both, same budget
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "data" / "ARC-AGI-3-Kaggle-Starter"
AGENTS = {
    "state": ROOT / "agi3" / "state_agent.py",
    "objects": ROOT / "agi3" / "object_agent.py",
    "warm": ROOT / "agi3" / "warm_agent.py",
    "explore": ROOT / "agi3" / "my_agent.py",
    "random": ROOT / "agi3" / "random_agent.py",
    # Frozen previous policy, kept so a change can be ablated against the thing
    # it replaced rather than only against the random floor.
    "argmax": ROOT / "agi3" / "argmax_agent.py",
    # Same policy as explore but EXPLORE_RATE=0.15, isolating the exploit
    # phase as the only variable. Currently the losing arm.
    "exploit": ROOT / "agi3" / "exploit_agent.py",
}
AB_ARMS = ("random", "exploit", "explore")

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


#: Appended to a copy of the starter's play_local.py so it also emits the
#: per-environment scores. Those carry the real formula -- see true_score() --
#: while the line play_local already prints is `Scorecard.score`, the max.
#: The anchor must be inside play_local's `main()`, where `sc` is in scope --
#: appending at module level fails with NameError because `arc` is a local.
PER_ENV_ANCHOR = '    print(f"\\nAggregate scorecard score: {score_val}")'

PER_ENV_PATCH = PER_ENV_ANCHOR + '''
    import json as _json
    _rows = []
    for _e in (getattr(sc, "environments", None) or []):
        _d = _e.model_dump() if hasattr(_e, "model_dump") else dict(_e)
        _rows.append({"keys": sorted(_d.keys()),
                      **{k: _d.get(k) for k in
                         ("id", "game_id", "score", "levels_completed", "actions")}})
    print("PER_ENV_JSON " + _json.dumps(_rows))'''

PER_ENV = re.compile(r"^PER_ENV_JSON (.*)$", re.M)


def true_score(rows: list[dict]) -> float:
    """Mean per-game score -- the metric the competition describes.

    Each game's score comes from the engine, which computes a level as

        ((human_baseline_actions / actions_taken) ** 2) * 100     capped at 100

    and then averages levels weighted by level index. Human baselines *are*
    available locally: all 25 environments expose `baseline_actions`.

    The only thing the library gets wrong is the final aggregation --
    `Scorecard.score` returns `max(...)` under a docstring saying average -- so
    we take the mean of the per-game scores it already computed correctly.

    Note the square. Clearing a level in 10x the human action count scores 1%
    of that level, not 10%.
    """
    if not rows:
        return 0.0
    return sum(float(r.get("score") or 0.0) for r in rows) / len(rows)


def patched_runner(agent: str) -> Path:
    """Write play_local.py + the per-env dump, leaving the original alone."""
    src = (STARTER / "scripts" / "play_local.py").read_text(encoding="utf-8")
    if PER_ENV_ANCHOR not in src:
        raise SystemExit("play_local.py changed shape; re-read it before patching")
    agent_anchor = 'ROOT / "agent" / "my_agent.py"'
    if agent_anchor not in src:
        raise SystemExit("play_local.py agent loader changed; inspect before running")
    src = src.replace(agent_anchor, 'Path(__import__("os").environ["ARC_AGENT_SOURCE"])')
    out = STARTER / "scripts" / f"_play_local_scored_{agent}.py"
    out.write_text(src.replace(PER_ENV_ANCHOR, PER_ENV_PATCH), encoding="utf-8")
    return out


def venv_python() -> Path:
    for cand in (STARTER / ".venv" / "Scripts" / "python.exe", STARTER / ".venv" / "bin" / "python"):
        if cand.exists():
            return cand
    raise SystemExit(f"No venv under {STARTER}. Run: bash scripts/setup.sh")


def play(agent: str, games: str, steps: int, seed: int = 0) -> dict:
    src = AGENTS[agent]
    if not src.exists():
        raise SystemExit(f"missing agent source {src}")
    runner = patched_runner(agent).relative_to(STARTER).as_posix()
    cmd = [str(venv_python()), runner, "--max-steps", str(steps)]
    if games:
        cmd += ["--game", games]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "ARC_SEED": str(seed),
           "PYTHONHASHSEED": "0", "ARC_AGENT_SOURCE": str(src.resolve())}
    # The child prints UTF-8; this box's locale is GBK, so decode explicitly.
    proc = subprocess.run(
        cmd, cwd=STARTER, capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )

    per_game, score, rows = {}, None, []
    for line in proc.stdout.splitlines():
        if m := PER_ENV.match(line):
            rows = json.loads(m.group(1))
            continue
        if m := SUMMARY.match(line):
            per_game[m.group(1)] = {
                "levels": int(m.group(2)),
                "actions": int(m.group(3)),
                "state": m.group(4),
            }
        elif m := SCORE.search(line):
            score = float(m.group(1))
    if proc.returncode != 0 or score is None or not per_game or not rows:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:])
        raise SystemExit(f"{agent}: incomplete run or missing per-game scorecard "
                         f"(exit {proc.returncode})")
    return {"score": score, "games": per_game, "true": true_score(rows), "rows": rows}


def report(label: str, res: dict) -> None:
    total = sum(g["levels"] for g in res["games"].values())
    print(f"\n--- {label} ---  TRUE {res['true']:.4f}"
          f"   progress {progress(res['games']):.4f}"
          f"   levels {total}"
          f"   lib-score(max) {res['score']:.4f}")
    by_id = {(r.get("game_id") or r.get("id") or "").split("-")[0]: r for r in res["rows"]}
    for gid, g in sorted(res["games"].items()):
        r = by_id.get(gid, {})
        sc = r.get("score")
        flag = "  <-- progress" if g["levels"] else ""
        print(f"    {gid:6} levels={g['levels']:2}  actions={g['actions']:5}"
              f"  score={sc if sc is None else f'{float(sc):.4f}'}{flag}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="explore", choices=list(AGENTS))
    ap.add_argument("--games", default="", help="comma-separated ids; empty = all available")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ab", action="store_true", help="compare random, exploit, and explore")
    ap.add_argument("--arms", nargs="+", choices=list(AGENTS),
                    help="explicit agents to compare under the same seeds")
    ap.add_argument("--out", type=Path, help="save per-game results after each run")
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat the whole A/B under N randomisations and report "
                         "the spread. One run is a noisy sample.")
    args = ap.parse_args()
    if args.seeds < 1 or args.steps < 1:
        ap.error("--seeds and --steps must be positive")

    def levels(res: dict) -> int:
        return sum(g["levels"] for g in res["games"].values())

    if args.ab or args.arms or args.seeds > 1 or args.out:
        arms = args.arms or (AB_ARMS if args.ab else (args.agent,))
        records = []
        rows: dict[str, list[tuple[int, float, float]]] = {a: [] for a in arms}
        cleared: dict[str, set[str]] = {a: set() for a in arms}
        for seed in range(args.seeds):
            for arm in arms:
                source_hash = hashlib.sha256(AGENTS[arm].read_bytes()).hexdigest()
                res = play(arm, args.games, args.steps, seed=seed)
                records.append({"arm": arm, "seed": seed, "steps": args.steps,
                                "source_sha256": source_hash, **res})
                if args.out:
                    args.out.parent.mkdir(parents=True, exist_ok=True)
                    args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
                rows[arm].append((levels(res), progress(res["games"]), res["true"]))
                cleared[arm] |= {g for g, v in res["games"].items() if v["levels"]}
                if args.seeds == 1:
                    report(f"{arm} ({args.steps} steps/game)", res)

        if args.seeds > 1:
            print(f"\n=== {args.seeds} seeds x {args.steps} steps/game ===")
            for arm in arms:
                lv = [r[0] for r in rows[arm]]
                pr = [r[1] for r in rows[arm]]
                sc = [r[2] for r in rows[arm]]
                print(f"  {arm:<8} levels {lv} mean {sum(lv)/len(lv):.1f}"
                      f" | progress mean {sum(pr)/len(pr):.4f}"
                      f" [{min(pr):.4f}-{max(pr):.4f}]"
                      f" | TRUE SCORE mean {sum(sc)/len(sc):.6f}"
                      f" [{min(sc):.6f}-{max(sc):.6f}]")
                print(f"           ever cleared: {', '.join(sorted(cleared[arm])) or 'none'}")
        else:
            print()
            for arm in arms:
                lv, pr, sc = rows[arm][0]
                print(f"  {arm:<8} levels {lv:2}  progress {pr:.4f}  TRUE {sc:.6f}")
    else:
        report(f"{args.agent} ({args.steps} steps/game)",
               play(args.agent, args.games, args.steps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
