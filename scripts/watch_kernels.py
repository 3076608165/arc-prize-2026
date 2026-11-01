import json, os, subprocess, sys, time
from datetime import datetime, timezone

os.environ.setdefault("NO_PROXY", "*")
os.environ.setdefault("no_proxy", "*")

OUT = r"D:/ARC Prize 2026/experiments/score-push-20261101"
os.makedirs(OUT, exist_ok=True)
LOG = os.path.join(OUT, "kernel-watch.jsonl")

KERNELS = {
    "agi3-stack-s16": "uts3125025522741/arc-agi-3-stack-s16",
    "agi2-triple-pass": "uts3125025522741/arc-agi-2-triple-pass",
}
DEADLINE = time.time() + 6 * 3600

def status(ref):
    r = subprocess.run(["kaggle", "kernels", "status", ref], capture_output=True, text=True, timeout=180)
    txt = (r.stdout + r.stderr).strip().splitlines()
    last = txt[-1] if txt else ""
    for tok in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if tok in last:
            return tok
    return "UNKNOWN:" + last[-90:]

state = {k: None for k in KERNELS}
with open(LOG, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"event": "watch-start", "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    while time.time() < DEADLINE:
        line = {"at": datetime.now(timezone.utc).isoformat(), "status": {}}
        done = 0
        for name, ref in KERNELS.items():
            try:
                s = status(ref)
            except Exception as e:
                s = "EXC:" + repr(e)[:80]
            line["status"][name] = s
            if s in ("COMPLETE", "ERROR", "CANCEL"):
                done += 1
        fh.write(json.dumps(line) + "\n")
        fh.flush()
        print(line, flush=True)
        if done == len(KERNELS):
            fh.write(json.dumps({"event": "all-terminal", "at": line["at"]}) + "\n")
            break
        time.sleep(90)
    else:
        fh.write(json.dumps({"event": "watch-timeout", "at": datetime.now(timezone.utc).isoformat()}) + "\n")
