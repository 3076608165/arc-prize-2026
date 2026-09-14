"""Push the queued ARC-AGI-2 experiment kernels as GPU sessions free up,
and log both pending submission scores.

Kaggle caps concurrent batch GPU sessions at 2, so the three-arm + base-model
plan could not start in parallel (push error: "Maximum batch GPU session count
of 2 reached"). This keeps at most MAX_SESSIONS kernels live, pushing the next
queued arm the moment one finishes, and stops when the queue is drained.

    python scripts/watch_exp3.py            # loops until done
    python scripts/watch_exp3.py --once     # single probe, for a manual check
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "watch-exp3.jsonl"
USER = "uts3125025522741"
MAX_SESSIONS = 2          # what Kaggle enforces, per the push error
POLL_SECONDS = 300

#: In order, with the prepared output directory to push. Each starts as soon as
#: a GPU session is free. `agi2/out/exp-m8b` (the 8B base-model swap) is prepared
#: and verified but deliberately NOT queued: it spends a fresh GPU session on an
#: unapproved hypothesis, so it waits for a decision. Append it to run it.
QUEUE = [("arc-agi-2-exp-aug24", "agi2/out/exp-aug24"),
         ("arc-agi-2-exp-ep2", "agi2/out/exp-ep2"),
         ("arc-agi-2-exp-inf4", "agi2/out/exp-inf4")]
KERNELS = [ref for ref, _ in QUEUE]
COMPETITIONS = ["arc-prize-2026-arc-agi-2", "arc-prize-2026-arc-agi-3"]
LIVE = ("RUNNING", "QUEUED")


def run(*args, cwd=None):
    try:
        p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                           timeout=180, cwd=cwd)
        return (p.stdout or "") + (p.stderr or "")
    except Exception as e:  # a network hiccup must not kill the watcher
        return f"ERROR {e!r}"


def kernel_status(ref):
    out = run("kernels", "status", f"{USER}/{ref}")
    for word in ("RUNNING", "COMPLETE", "ERROR", "QUEUED", "CANCEL"):
        if word in out:
            return word
    return "NOT_PUSHED" if "404" in out else f"UNKNOWN({out.strip()[:60]})"


def pending_scores():
    """{competition: [{ref, status, public}]} for the two newest submissions.

    Anchors on the status token, not column offsets: the description column
    contains spaces, so splitting by width is unreliable. Statuses come from the
    CLI as SubmissionStatus.<NAME>.
    """
    seen = {}
    for comp in COMPETITIONS:
        rows = []
        for line in run("competitions", "submissions", "-c", comp).splitlines()[1:3]:
            parts = line.split()
            idx = next((i for i, p in enumerate(parts)
                        if p.startswith("SubmissionStatus.")), None)
            if not parts or idx is None:
                continue
            rows.append({"ref": parts[0],
                         "status": parts[idx].split(".", 1)[1],
                         "public": parts[idx + 1] if idx + 1 < len(parts) else ""})
        seen[comp] = rows
    return seen


def push(out_dir):
    out = run("kernels", "push", "-p", ".", cwd=str(ROOT / out_dir))
    lines = [l for l in out.strip().splitlines() if l.strip()]
    return "uccessfully" in out, (lines[-1][:160] if lines else "")


def main():
    once = "--once" in sys.argv
    #: refs this process has successfully launched, so a tick does not re-push
    #: an arm that is still queueing on Kaggle's side.
    pushed = set()
    while True:
        status = {k: kernel_status(k) for k in KERNELS}
        row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "kernels": status, "submissions": pending_scores()}
        live = [k for k, v in status.items() if v in LIVE]

        # Start the next queued arm only if a session is actually free. Two
        # pushes in one tick would hit the cap and one would silently fail.
        launches = []
        for ref, out_dir in QUEUE:
            if len(live) + len(launches) >= MAX_SESSIONS:
                break
            if status[ref] in ("NOT_PUSHED", "ERROR") and ref not in pushed:
                ok, msg = push(out_dir)
                launches.append({"ref": ref, "ok": ok, "msg": msg})
                if ok:
                    pushed.add(ref)
                    live.append(ref)
        if launches:
            row["pushed"] = launches
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

        if once:
            return 0
        if not live and all(status[r] in ("COMPLETE", "RUNNING") for r in KERNELS):
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
