"""Fire today's declared draws, once each, retrying only where the write provably did not land.

Both ARC boards allow one submission per day and count the best two across the
whole competition, so an unused slot is simply lost. The trap this script exists
to avoid: Kaggle's CLI prints "N submissions remaining today." on *success*, so
its text cannot tell a refusal from an acceptance -- reading that line as a
refusal both invents a quota outage and hides a submission that already
happened. Acceptance is therefore decided by reading the submission list back,
and a retry is issued only when the description is absent from it.

    python scripts/fire_draws.py --targets scripts/fire-draws-20260915.json
    python scripts/fire_draws.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
READBACK_TRIES = 3
READBACK_WAIT = 10


def kaggle(*args: str) -> str:
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", *args],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def verdict(raised: bool, in_list: bool) -> str:
    """Acceptance is the server's list, never the CLI's text.

    Reads only the list: "submitted" (list has it, whatever the CLI said),
    "retry" (not listed and the call failed), "uncertain" (not listed but the
    call succeeded, which must never be retried blind).
    """
    if in_list:
        return "submitted"
    return "retry" if raised else "uncertain"


def submissions(competition: str) -> list[dict]:
    return list(csv.DictReader(io.StringIO(
        kaggle("competitions", "submissions", "-c", competition, "--csv"))))


def listed(target: dict, tries: int = 1) -> dict | None:
    for attempt in range(tries):
        if attempt:
            time.sleep(READBACK_WAIT)
        row = next((r for r in submissions(target["competition"])
                    if r.get("description") == target["message"]), None)
        if row is not None:
            return row
    return None


def kernel_ref(kernel: str, version: int | None = None) -> str:
    """Pin the readiness check to the arm's own version.

    Arms live as successive versions of one kernel, so checking the bare ref would
    gate today's arm on whatever *other* arm happens to be running today.
    """
    return f"{kernel}/{version}" if version else kernel


def kernel_ready(kernel: str, version: int | None = None) -> bool:
    status = kaggle("kernels", "status", kernel_ref(kernel, version))
    if any(word in status for word in ("ERROR", "CANCEL", "FAILED")):
        raise RuntimeError(f"kernel will never be submittable: {status}")
    return "KernelWorkerStatus.COMPLETE" in status


def fire(target: dict, log, poll: int, quota_wait: int, deadline: float,
         keep_trying: bool) -> str:
    if listed(target) is not None:
        log({"event": "already_submitted", **target})
        return "already_submitted"

    while not kernel_ready(target["kernel"], target.get("version")):
        if time.monotonic() > deadline:
            log({"event": "deadline", "stage": "kernel", **target})
            return "deadline"
        time.sleep(poll)

    while True:
        raised = None
        try:
            kaggle("competitions", "submit", target["competition"],
                   "-k", target["kernel"], "-v", str(target["version"]),
                   "-f", target["file"], "-m", target["message"])
        except RuntimeError as exc:
            raised = str(exc)
        row = listed(target, READBACK_TRIES)
        outcome = verdict(raised is not None, row is not None)
        log({"event": outcome, "error": raised, "row": row, **target})
        if outcome == "submitted":
            return "submitted"
        if outcome == "uncertain":
            return "uncertain"
        if not keep_trying or time.monotonic() > deadline:
            log({"event": "deadline", "stage": "quota", **target})
            return "deadline"
        time.sleep(quota_wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--log", type=Path,
                        default=ROOT / "experiments" / "fire-draws.jsonl")
    parser.add_argument("--poll", type=int, default=60)
    parser.add_argument("--quota-wait", type=int, default=600)
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--retry", action="store_true",
                        help="re-attempt a target the server has not listed (the "
                             "CLI text cannot justify this; the list can)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        # Every case that the CLI's text gets wrong.
        assert verdict(raised=True, in_list=True) == "submitted"   # success printed as an error
        assert verdict(raised=False, in_list=False) == "uncertain"  # silent success
        assert verdict(raised=True, in_list=False) == "retry"
        assert verdict(raised=False, in_list=True) == "submitted"
        assert kernel_ref("owner/nb", 12) == "owner/nb/12"   # the arm's own version
        assert kernel_ref("owner/nb", None) == "owner/nb"
        print("self-test ok: the list decides, the text never does")
        return 0
    if args.targets is None:
        parser.error("--targets is required")

    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    deadline = time.monotonic() + args.hours * 3600
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as handle:
        def log(record: dict) -> None:
            record["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            print(json.dumps(record), flush=True)
        for target in targets:
            if fire(target, log, args.poll, args.quota_wait, deadline,
                    args.retry) == "uncertain":
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
