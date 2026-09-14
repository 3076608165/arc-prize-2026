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
import calendar
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from watch_exp3 import pending_scores  # reuse the status-token parser, do not re-derive it

LOG = ROOT / "experiments" / "watch-exp3.jsonl"
#: 5 min was inherited from watch_exp3, where it had to catch a free GPU session
#: and launch the next arm in the same tick. Nothing here is latency-sensitive:
#: a 12h competition rerun is being waited on, and the next submission slot does
#: not open until 00:00Z, so 15 min costs ~5 min of staleness on an answer that
#: took 8.5h to arrive and saves two thirds of the CLI calls. Lower it only if
#: something downstream starts gating on the minute.
POLL_SECONDS = 900


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

    #: The submission's own start time and the competition's runtime cap bound
    #: when a result can first exist: a run that started at T cannot have scored
    #: before T + 720 min. Polling inside that window is provably wasted calls,
    #: so wait it out in one sleep and spend the polls after it.
    if "--not-before" in sys.argv and not once:
        at = sys.argv[sys.argv.index("--not-before") + 1]
        delay = calendar.timegm(time.strptime(at, "%Y-%m-%dT%H:%M:%SZ")) - time.time()
        if delay > 0:
            print(f"sleeping {delay / 60:.0f} min: no result can exist before {at}", flush=True)
            time.sleep(delay)

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
