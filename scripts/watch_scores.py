"""Collect server results for this scoring round; never submit or retry a write."""
from __future__ import annotations

import argparse
import csv
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from submit_when_ready import kaggle

TARGETS = {
    "agi2": ("arc-prize-2026-arc-agi-2",
             "20260910 augment8 v10: reduce TTT work; full 11.83h competition budget"),
    "agi3": ("arc-prize-2026-arc-agi-3",
             "20260910 objects v3: visible-object resampling, stable game seed; public25 0.315 to 0.452"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("experiments/score-push-20260910.json"))
    parser.add_argument("--hours", type=float, default=14)
    args = parser.parse_args()
    if args.hours <= 0:
        parser.error("--hours must be positive")
    report = json.loads(args.report.read_text(encoding="utf-8"))
    deadline = time.monotonic() + args.hours * 3600
    previous = {}
    terminal = set()
    while time.monotonic() < deadline:
        for track, (competition, message) in TARGETS.items():
            if track in terminal:
                continue
            try:
                rows = csv.DictReader(io.StringIO(kaggle(
                    "competitions", "submissions", "-c", competition, "--csv")))
                row = next((r for r in rows if r.get("description") == message), None)
                if row is None:
                    status = "WAITING_FOR_SUBMISSION"
                else:
                    status = row["status"].rsplit(".", 1)[-1]
                    report[track]["server_submission"] = row
                    report[track]["submission_id"] = int(row["ref"])
                    if row.get("publicScore"):
                        report[track]["public_score"] = float(row["publicScore"])
                    if status in {"COMPLETE", "ERROR", "FAILED", "CANCELLED"}:
                        terminal.add(track)
                report[track]["leaderboard_status"] = status
                report[track].pop("poll_error", None)
            except Exception as exc:
                status = "POLL_ERROR"
                report[track]["poll_error"] = str(exc)
            summary = (status, report[track].get("public_score"))
            if previous.get(track) != summary:
                print(json.dumps({"track": track, "status": status,
                                  "public_score": summary[1]}), flush=True)
                previous[track] = summary
        report["last_checked_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = args.report.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.report)
        if len(terminal) == len(TARGETS):
            return
        time.sleep(45)
    print("Score collection timed out; see report for last confirmed server state.", flush=True)


if __name__ == "__main__":
    main()
