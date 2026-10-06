"""Wait for a pushed kernel's commit run to finish, pull its smoke, attribute it.

The commit run is the only CLI-readable session (`kernels output` returns the newest
one), and it is the free instrument every config/agent knob is read through, so this
is the three steps the 2026-10-06 round did by hand:

    python scripts/read_smoke.py agi3/out/duck-sota-v3            # wait, pull, attribute
    python scripts/read_smoke.py agi3/out/duck-sota-v3 --once     # one status probe

Writes <out>/runout/ (+ token-attribution.json inside it) and prints the reading against
the v1 baseline: aggregate tok/s, games served, completion tokens per request, mean score.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KERNEL = "uts3125025522741/arc-agi-3-duck-sota-open-source-lift"
BASELINE = {"aggregate_tok_s": 556.3, "games_served": "10 of 25",
            "completion_per_request_mean": 1241.1, "mean_public_score": 0.4589}


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "agi3/out/duck-sota-v3")
    once = "--once" in sys.argv
    a = api()
    while True:
        status = a.kernels_status(KERNEL).get("status")
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {KERNEL} -> {status}", flush=True)
        if status == "COMPLETE":
            break
        if status == "ERROR":
            print("commit run failed; nothing to attribute", flush=True)
            return 1
        if once:
            return 0
        time.sleep(120)

    runout = ROOT / out / "runout"
    runout.mkdir(parents=True, exist_ok=True)
    a.kernels_output(KERNEL, path=str(runout), force=True)
    done = subprocess.run([sys.executable, "-X", "utf8",
                           str(ROOT / "scripts" / "smoke_token_attribution.py"), str(runout)])
    if done.returncode != 0:
        return done.returncode
    got = json.loads((runout / "token-attribution.json").read_text(encoding="utf-8"))
    print(json.dumps({"arm": out.name, "baseline": BASELINE, "measured": {
        "requests": got["requests"],
        "completion_per_request_mean": got["completion_per_request"]["mean"],
        "completion_tokens": got["completion_tokens"],
        "actions": got["actions"],
        "games_served": f"{got['games_with_any_action']} of {got['games_total']}",
        "mean_public_score": got["mean_public_score"],
        "generated_tokens_bench": got["generated_tokens_bench"],
    }}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
