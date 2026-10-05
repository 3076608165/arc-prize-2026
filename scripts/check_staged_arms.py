"""What would the two scheduled invocations fire on the coming UTC days?

`fire_next_arm.py --dry-run` answers that for TODAY. This answers it for the day
the arms staged in this session are meant to cover, by reading the same live board
with `day` pinned forward. A queue whose head is already fired, or whose board is
already spent, is the silent no-op that lost 2026-10-01..03 while every cron run
reported success -- so the staged arms are verified reachable BEFORE the cron runs.

    python scripts/check_staged_arms.py 2026-10-06 [2026-10-07 ...]

Needs an interpreter that can import `kaggle` (the repo has no venv):
    ~/AppData/Local/hermes/hermes-agent/venv/Scripts/python.exe

Caveat: the projection is only meaningful for the NEXT day. It does not simulate
fires it is predicting, so running it for a second day re-prints the same arms --
read day N+1 after day N has actually fired.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fire_next_arm as f            # noqa: E402
from fire_draws import submissions   # noqa: E402

QUEUE = Path(__file__).resolve().parent / "arm-queue.json"


def main(days):
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    competitions = {arm["competition"] for arm in queue}
    for day in days:
        try:
            desc, spent = f.board_state(competitions, day=day)
        except RuntimeError as exc:                     # board unreachable
            print(f"{day}: BOARD READ FAILED -- {exc}")
            continue
        # The two scheduled invocations in order: each fires one arm and spends a board.
        a = f.select(queue, desc, spent)
        spent2 = spent | {a["competition"]} if a else spent
        b = f.select(queue, desc | ({a["message"]} if a else set()), spent2)
        for label, arm in (("inv1", a), ("inv2", b)):
            print(f"{day} {label}: {arm['id']:22s} {arm['competition'] if arm else '-'}"
                  if arm else f"{day} {label}: NOTHING TO FIRE (queue drained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or [time.strftime("%Y-%m-%d", time.gmtime())]))
