"""Fire the next un-fired arm from the queue, whatever time it is.

One slot per board per UTC day, and a slot is valid for the whole UTC day, so the
only thing a fixed firing time buys is earliness -- never correctness. That is why
this is a *queue advance* rather than a clock event: run it any time after 00:00
UTC and it fires the same arm a 10:05 run would have fired. Run it late, run it
twice, run it after the machine was off for three days -- the arm's own message is
the key, so a repeat run re-reads the submission list, finds it, and does nothing
(`fire_draws.py` decides acceptance by reading the list back, never by the CLI's
text).

Catch-up is deliberately one arm per invocation: firing a missed arm and the
current one on the same day is impossible (one slot), so the queue advances by
exactly one and the rest wait for their own days.

    python scripts/fire_next_arm.py --dry-run     # what would fire, and why
    python scripts/fire_next_arm.py               # push the queue by one
    python scripts/fire_next_arm.py --self-test   # queue-selection logic
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "scripts" / "arm-queue.json"
LOG = ROOT / "experiments" / "fire-arms.jsonl"


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


def select(queue: list[dict], fired: set[str]) -> dict | None:
    """First queued arm whose message has not been accepted by the server yet."""
    return next((arm for arm in queue if arm["message"] not in fired), None)


def arm_targets(arm: dict) -> list[dict]:
    keys = ("competition", "kernel", "version", "file", "message")
    return [{k: arm[k] for k in keys}]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=QUEUE)
    parser.add_argument("--log", type=Path, default=LOG)
    parser.add_argument("--hours", type=float, default=6.0,
                        help="deadline for kernel-ready + quota retries")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        q = [{"message": "a"}, {"message": "b"}, {"message": "c"}]
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
        print("self-test ok: the queue advances in order and never re-fires")
        return 0

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    fired = fired_messages(args.log)
    arm = select(queue, fired)
    if arm is None:
        print(json.dumps({"event": "queue_empty", "queued": len(queue),
                          "fired": len(fired)}))
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
