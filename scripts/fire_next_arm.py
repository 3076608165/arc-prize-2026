"""Fire the next un-fired arm from the queue, whatever time it is.

One slot per board per UTC day, and a slot is valid for the whole UTC day, so the
only thing a fixed firing time buys is earliness -- never correctness. That is why
this is a *queue advance* rather than a clock event: run it any time after 00:00
UTC and it fires the same arm a 10:05 run would have fired. Run it late, run it
twice, run it after the machine was off for three days -- the arm's own message is
the key, so a repeat run re-reads the submission list, finds it, and does nothing
(`fire_draws.py` decides acceptance by reading the list back, never by the CLI's
text).

"Already fired" is therefore read from the SERVER, not from this script's own log.
An arm fired by a hand-run round (`scripts/fire-draws-<date>.json`, logged under
`experiments/score-push-<date>/fire.jsonl`) leaves the local log untouched, so a
log-only queue sees its head as un-fired, re-selects it, and spends the invocation
re-reading a submission that already exists -- one whole UTC day lost per stale
head. The submission list is the authority on what has been fired.


Catch-up is deliberately one arm per invocation: firing a missed arm and the
current one for the SAME board on the same day is impossible (one slot), so each
invocation advances the queue by one and the rest wait for their own days. Two
boards have daily slots, so the invocation is scheduled twice (10:05 and 10:25
local) and the queue order allocates them -- which only works if the two
invocations are told apart by TIMING, and a catch-up burst gives them the same
instant. Measured 2026-09-27: both routes dispatched at 14:08:04 local, both read
the board before either write landed, both selected the AGI-2 head, and the
AGI-3 slot was left unspent. Two guards, both needed:

  * `exclusive()` -- one invocation inside read-decide-fire at a time, so the
    second one reads a board that already contains the first one's submission.
  * a board that already has a submission dated today is skipped outright, so
    however many routes fire (one invocation, four, or one that runs hours late)
    each board still spends exactly its one slot.

    python scripts/fire_next_arm.py --dry-run     # what would fire, and why
    python scripts/fire_next_arm.py               # push the queue by one
    python scripts/fire_next_arm.py --self-test   # selection + lock logic
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fire_draws import submissions   # sits beside us; always run as a script

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "arm-queue.json"
LOG = ROOT / "experiments" / "fire-arms.jsonl"
LOCK = ROOT / "scripts" / ".fire-next-arm.lock"


def fired_messages(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    out = set()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") in ("submitted", "already_submitted") and rec.get("message"):
            out.add(rec["message"])
    return out


def select(queue: list[dict], fired: set[str], spent: set[str] | None = None) -> dict | None:
    """First queued arm whose message has not been accepted by the server yet and
    whose board still has today's slot."""
    spent = spent or set()
    return next((arm for arm in queue
                 if arm["message"] not in fired and arm["competition"] not in spent), None)


def board_state(competitions: set[str], day: str | None = None,
                rows=submissions) -> tuple[set[str], set[str]]:
    """(every submission description already on the server, boards spent today).

    The local log only knows about arms this repo's own scripts fired. A round
    fired by hand (or by another agent, or by the second cron route) is invisible
    to it, so a log-only queue re-selects its stale head, no-ops, and loses the
    day -- which is exactly what happened on 2026-09-25. `rows` is injectable so
    the merge is testable without the network.

    The spend check reads the same rows for free: the board's own date column is
    UTC, and a submission dated today means that board's slot is gone whatever
    arm (or route) spent it.
    """
    day = day or time.strftime("%Y-%m-%d", time.gmtime())
    descriptions: set[str] = set()
    spent: set[str] = set()
    for competition in sorted(competitions):
        for row in rows(competition):
            if row.get("description"):
                descriptions.add(row["description"])
            if str(row.get("date") or "").startswith(day):
                spent.add(competition)
    return descriptions, spent


def _try_lock(handle) -> None:
    """Take the lock or raise OSError. The OS drops it when the holder dies, so
    there is no stale-lock timeout to maintain -- only a bounded wait."""
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextlib.contextmanager
def exclusive(timeout: float, poll: float = 5.0):
    """Serialize invocations across the read-decide-fire section.

    All four routes (two Hermes cron jobs, two Windows tasks) catch up in one
    burst after the machine has been asleep, so the twenty minutes that is
    supposed to separate them does not exist and every route selects the same
    arm. Waiting is safe: the holder only blocks while it waits for a kernel
    that is not ready yet, and a wait that expires is reported rather than
    silently skipped.
    """
    handle = LOCK.open("a+")
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                _try_lock(handle)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"another invocation has held {LOCK.name} "
                                       f"for more than {timeout:g}s")
                time.sleep(poll)
        yield
    finally:
        handle.close()


def arm_targets(arm: dict) -> list[dict]:
    keys = ("competition", "kernel", "version", "file", "message")
    return [{k: arm[k] for k in keys}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--hours", type=float, default=6.0,
                        help="deadline for kernel-ready + quota retries")
    parser.add_argument("--lock-wait", type=float, default=4 * 3600.0,
                        help="seconds to wait for a concurrent invocation to "
                             "finish before reporting the burst as contended")
    parser.add_argument("--hold-lock", type=float, default=0,
                        help="hold the fire lock for N seconds, then exit (the "
                             "lock's own check; fires nothing)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.hold_lock:
        with exclusive(args.lock_wait):
            print(json.dumps({"event": "lock_held", "seconds": args.hold_lock}))
            time.sleep(args.hold_lock)
        return 0

    if args.self_test:
        q = [{"message": "a", "competition": "arc-a"},
             {"message": "b", "competition": "arc-a"},
             {"message": "c", "competition": "arc-b"}]
        assert select(q, set())["message"] == "a"          # fresh queue
        assert select(q, {"a"})["message"] == "b"          # next in order
        assert select(q, {"a", "c"})["message"] == "b"     # order, not count
        assert select(q, {"a", "b", "c"}) is None          # empty
        assert select([], set()) is None
        assert fired_messages(Path("does-not-exist.jsonl")) == set()
        # Round-trip through a real log file: the queue is advanced by what the
        # server accepted, so parsing the log wrong would re-fire an arm.
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "fire-arms.jsonl"
            log.write_text("\n".join([
                json.dumps({"event": "already_submitted", "message": "a"}),
                "{not json}",
                json.dumps({"event": "deadline", "message": "b"}),
                json.dumps({"event": "submitted", "message": "b"}),
            ]), encoding="utf-8")
            assert fired_messages(log) == {"a", "b"}
            assert select(q, fired_messages(log))["message"] == "c"
        # An arm fired by a hand-run round exists only on the SERVER. A log-only
        # queue re-selects that stale head and spends the day re-reading it.
        rows = lambda competition: [{"description": "a"}, {"description": "b"}]
        assert board_state({"arc-a"}, rows=rows) == ({"a", "b"}, set())
        assert board_state({"arc-a", "arc-b"}, rows=rows) == ({"a", "b"}, set())
        assert board_state({"arc-a"}, rows=lambda c: [{}, {"description": ""}]) == (set(), set())
        assert select(q, board_state({"arc-a"}, rows=rows)[0])["message"] == "c"
        # The two routes are dispatched in the same catch-up burst when the
        # machine was asleep (2026-09-27). Log-and-board dedup alone is not
        # enough then: the head is un-fired and its board's slot is already
        # spent, so it must be skipped and the OTHER board's arm selected --
        # otherwise the burst spends two invocations on one board and leaves the
        # other board's slot unfired.
        spent_rows = lambda competition: (
            [{"description": "a", "date": "2026-09-27 04:08:12.000000"}]
            if competition == "arc-a" else [{"description": "", "date": ""}])
        descriptions, spent = board_state({"arc-a", "arc-b"}, day="2026-09-27",
                                          rows=spent_rows)
        assert spent == {"arc-a"} and descriptions == {"a"}
        burst = [{"message": "a", "competition": "arc-a"},
                 {"message": "harness", "competition": "arc-b"}]
        assert select(burst, descriptions, spent)["message"] == "harness"
        # Yesterday's submission does not spend today's slot.
        old = lambda competition: [{"description": "a", "date": "2026-09-26 05:52:11.000000"}]
        assert board_state({"arc-a"}, day="2026-09-27", rows=old)[1] == set()
        # The lock is per-process state on Windows, so it has to be checked
        # against a real second process: a same-process check would pass even if
        # locking did nothing.
        holder = subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                   "--hold-lock", "6"])
        try:
            deadline = time.monotonic() + 30
            while holder.poll() is None and time.monotonic() < deadline:
                try:
                    with exclusive(0.1, poll=0.1):
                        pass
                except TimeoutError:
                    break                       # held, as it must be
                time.sleep(0.2)
            else:
                raise AssertionError("the lock was not held by the other process")
            holder.wait(timeout=60)
            with exclusive(15, poll=0.5):        # released when the holder exited
                pass
        finally:
            holder.kill()
        print("self-test ok: the queue advances in order, never re-fires, and "
              "the two routes cannot select the same arm")
        return 0

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    try:
        with exclusive(args.lock_wait):
            return advance(queue, args)
    except TimeoutError as exc:
        print(json.dumps({"event": "lock_timeout", "error": str(exc)}))
        return 3


def advance(queue: list[dict], args) -> int:
    """Read the board, pick the arm, fire it. Caller holds the lock."""
    fired = fired_messages(args.log)
    try:
        descriptions, spent = board_state({arm["competition"] for arm in queue})
    except RuntimeError as exc:
        # Deliberately NOT a fallback to the log-only set: that silent no-op is the
        # day-losing bug this read exists to remove. Fail loudly, exit nonzero, let
        # the caller's retry bring it back.
        print(json.dumps({"event": "board_read_failed", "error": str(exc)}))
        return 2
    arm = select(queue, fired | descriptions, spent)
    if arm is None:
        print(json.dumps({"event": "queue_empty", "queued": len(queue),
                          "fired": len(fired | descriptions),
                          "spent_today": sorted(spent)}))
        return 0

    target_file = (ROOT / "experiments" / f"arm-{arm['id']}-targets.json")
    target_file.write_text(json.dumps(arm_targets(arm), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"event": "selected", "arm": arm["id"],
                      "version": arm["version"], "targets": str(target_file),
                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    if args.dry_run:
        return 0

    done = subprocess.run(
        [sys.executable, "-X", "utf8", str(ROOT / "scripts" / "fire_draws.py"),
         "--targets", str(target_file), "--log", str(args.log),
         "--hours", str(args.hours)])
    return done.returncode


if __name__ == "__main__":
    raise SystemExit(main())
