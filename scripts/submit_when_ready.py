"""Wait for a Kaggle commit, then submit one explicitly pinned version once."""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
import time
from pathlib import Path


def kaggle(*args: str) -> str:
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", *args],
                            capture_output=True, text=True, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--version", required=True, type=int)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.version < 1 or args.timeout < 1:
        parser.error("version and timeout must be positive")

    record = {"kernel": args.kernel, "version": args.version,
              "competition": args.competition, "message": args.message}

    def save(status: str, **details: object) -> None:
        record.update(status=status, **details)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": status, **details}), flush=True)

    # Do not automatically retry a previous attempt with an unknown outcome.
    if args.receipt.exists():
        previous = json.loads(args.receipt.read_text(encoding="utf-8"))
        if previous.get("status") in ("submitting", "submitted", "submission_uncertain"):
            raise SystemExit("Previous submission attempt exists; inspect Kaggle before retrying.")

    started = time.monotonic()
    previous_status = None
    while time.monotonic() - started < args.timeout:
        status = kaggle("kernels", "status", args.kernel)
        if status != previous_status:
            save("waiting", kernel_status=status)
            previous_status = status
        if 'KernelWorkerStatus.COMPLETE' in status:
            break
        if any(value in status for value in ("ERROR", "CANCEL", "FAILED")):
            save("kernel_failed", kernel_status=status)
            raise SystemExit(1)
        time.sleep(45)
    else:
        save("timeout")
        raise SystemExit(1)

    # A unique description makes restarting the monitor safe after completion.
    existing = csv.DictReader(io.StringIO(kaggle(
        "competitions", "submissions", "-c", args.competition, "--csv")))
    for row in existing:
        if row.get("description") == args.message:
            save("already_submitted", submission=row)
            return

    save("submitting")
    try:
        response = kaggle("competitions", "submit", args.competition,
                          "-k", args.kernel, "-v", str(args.version),
                          "-f", args.file, "-m", args.message)
    except Exception as exc:
        save("submission_uncertain", error=str(exc))
        raise
    # CLI success text may only report remaining quota. Confirm acceptance by
    # reading the server's submission list; never retry an uncertain write.
    try:
        rows = csv.DictReader(io.StringIO(kaggle(
            "competitions", "submissions", "-c", args.competition, "--csv")))
        accepted = next((row for row in rows if row.get("description") == args.message), None)
        if accepted is None:
            raise RuntimeError("Submission not yet visible in server history; inspect before retrying.")
    except Exception as exc:
        save("submission_uncertain", response=response, error=str(exc))
        raise
    save("submitted", response=response, submission=accepted)


if __name__ == "__main__":
    main()
