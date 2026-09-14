"""Wait for one PENDING Kaggle submission to score, then write the reading down.

Round-scoped watchers die with the shell that started them: that is how
`experiments/watch-exp3.jsonl` came to end frozen on PENDING with the answer
never recorded. This one is scoped to a single submission ref, survives nothing
in particular, and exits the moment the board reports a terminal status -- the
number is on disk either way.

    python scripts/wait_for_score.py 56223660 --once          # single probe
    python scripts/wait_for_score.py 56223660                 # poll until scored
    python scripts/wait_for_score.py 56223681 --comp arc-prize-2026-arc-agi-3
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from watch_exp3 import pending_scores  # reuse the status-token parser, do not re-derive it

LOG = ROOT / "experiments" / "watch-exp3.jsonl"
POLL_SECONDS = 300


def find(ref, comp):
    """The board row for `ref`, or None when the CLI did not list it."""
    for row in pending_scores().get(comp, []):
        if row["ref"] == ref:
            return row
    return None


def main():
    ref = sys.argv[1]
    comp = (sys.argv[sys.argv.index("--comp") + 1] if "--comp" in sys.argv
            else "arc-prize-2026-arc-agi-2")
    once = "--once" in sys.argv
    while True:
        # ponytail: only the two newest rows per competition are read, which is
        # why NOT_LISTED keeps polling instead of exiting -- the next daily
        # submission pushes this ref out of the window.
        row = find(ref, comp) or {"ref": ref, "status": "NOT_LISTED", "public": ""}
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = json.dumps({"t": stamp, "waiting": row})
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line, flush=True)

        if row["status"] not in ("PENDING", "NOT_LISTED"):
            out = ROOT / "experiments" / f"{ref}-result.json"
            out.write_text(json.dumps({"competition": comp,
                                       "observed_utc": stamp, **row}, indent=2) + "\n",
                           encoding="utf-8")
            return 0
        if once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
