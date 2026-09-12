"""Collect this fixed AGI-2 pair once, then compare; never submit or retry a push."""
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "experiments/agi2-coverage-v1"


def cli(*arguments):
    result = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", *arguments],
                            capture_output=True, text=True, encoding="utf-8", timeout=180)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
    return result.stdout.strip()


def main():
    DEST.mkdir(exist_ok=True)
    arms = {str(n): {"kernel": f"uts3125025522741/arc-agi-2-coverage-{n}",
                     "version": 1, "status": "uploaded"} for n in (16, 8)}
    previous = {}
    for _ in range(160):
        for arm, row in arms.items():
            if row["status"] in ("downloaded", "failed"):
                continue
            try:
                status = cli("kernels", "status", row["kernel"])
                row["server_status"] = status
                if "KernelWorkerStatus.COMPLETE" in status:
                    directory = DEST / f"augment{arm}"
                    row["download_response"] = cli("kernels", "output", row["kernel"] + "/1",
                        "-p", str(directory), "--file-pattern", r"(coverage_audit/.*|coverage_summary\.json|submission\.json|.*\.log)$")
                    row["status"] = "downloaded"
                elif any(value in status for value in ("ERROR", "CANCEL", "FAILED")):
                    row["status"] = "failed"
                    row["download_response"] = cli("kernels", "output", row["kernel"] + "/1",
                        "-p", str(DEST / f"augment{arm}"))
                else:
                    row["status"] = "running_or_queued"
                row.pop("poll_error", None)
            except Exception as exc:
                row["poll_error"] = str(exc)
            summary = (row["status"], row.get("server_status"), row.get("poll_error"))
            if previous.get(arm) != summary:
                print(json.dumps({"arm": arm, **row}), flush=True)
                previous[arm] = summary
        record = {"observed_utc": datetime.now(timezone.utc).isoformat(), "arms": arms}
        (DEST / "execution-status.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if all(row["status"] in ("downloaded", "failed") for row in arms.values()):
            if all(row["status"] == "downloaded" for row in arms.values()):
                subprocess.run([sys.executable, str(ROOT / "agi2/compare_coverage.py"),
                    str(DEST / "augment16"), str(DEST / "augment8"),
                    "--out", str(DEST / "comparison.json")], cwd=ROOT, check=True)
            return
        time.sleep(45)
    raise SystemExit("Collection timeout; inspect saved state, do not repush automatically.")


if __name__ == "__main__":
    main()
