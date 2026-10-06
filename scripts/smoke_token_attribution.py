"""Attribute a lift smoke's token bill from its own request logs.

The scored rerun gives each game `aggregate tok/s x 540 min / 110 games` tokens,
so the only per-arm quantities that matter are completion tokens per request and
the public-25 mean score. Both live in the pulled runout:

    python scripts/smoke_token_attribution.py agi3/out/duck-sota-v2/runout

Reads `*_p0_requests.jsonl` (event=response rows carry `usage`) and
`benchmark.json` (per-game actions, tokens, scores, levels), writes
`<runout>/token-attribution.json` and prints the summary. Bash-equivalent of the
2026-10-06 v2 reading: 125 requests, 141,548 completion tokens, mean 1,132.
"""
from __future__ import annotations

import collections
import glob
import json
import statistics
import sys
from pathlib import Path


def attribute(runout: Path) -> dict:
    rows = []
    for f in sorted(glob.glob(str(runout / "*_requests.jsonl"))):
        game = Path(f).name.split("_")[0]
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("event") != "response" or not d.get("usage"):
                continue
            rows.append({"game": game, "step": d.get("analysis_step"),
                         "ri": d.get("request_index_within_turn"), "fr": d.get("finish_reason"),
                         "pt": int(d["usage"].get("prompt_tokens") or 0),
                         "ct": int(d["usage"].get("completion_tokens") or 0)})
    if not rows:                                   # a smoke with no served game
        return {"requests": 0, "note": "no response rows in any *_requests.jsonl"}
    ct = sorted(r["ct"] for r in rows)
    per_game: collections.Counter = collections.Counter()
    for r in rows:
        per_game[r["game"]] += r["ct"]
    bench = json.loads((runout / "benchmark.json").read_text(encoding="utf-8"))
    runs = bench.get("game_runs", [])
    return {
        "requests": len(rows),
        "games_with_requests": len({r["game"] for r in rows}),
        "completion_tokens": sum(ct),
        "prompt_tokens": sum(r["pt"] for r in rows),
        "completion_per_request": {
            "mean": round(sum(ct) / len(ct), 1), "p50": int(statistics.median(ct)),
            "p90": ct[int(0.9 * len(ct))], "max": ct[-1]},
        "requests_per_turn": dict(collections.Counter("ri=%s" % r["ri"] for r in rows)),
        "finish_reasons": dict(collections.Counter(r["fr"] for r in rows)),
        "tokens_per_game": dict(per_game.most_common()),
        "games_with_any_action": sum(1 for g in runs if g.get("history")),
        "games_total": len(runs),
        "actions": sum(len(g.get("history") or []) for g in runs),
        "generated_tokens_bench": sum(g.get("final_generated_tokens") or 0 for g in runs),
        "mean_public_score": round(sum(g.get("final_score") or 0.0 for g in runs) / max(1, len(runs)), 4),
        "levels_cleared": {g["game_id"]: [g.get("levels_completed"), g.get("number_of_levels")]
                           for g in runs if g.get("levels_completed")},
    }


def main() -> int:
    runout = Path(sys.argv[1] if len(sys.argv) > 1 else "agi3/out/duck-sota-v2/runout")
    out = attribute(runout)
    (runout / "token-attribution.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
