"""Print the machine-readable competition constraints for the three ARC 2026 tracks.

They live in Kaggle's internal API, not in the rendered page, which is JS-only:

    competitions.CompetitionService/GetCompetition {"competitionId": <id>}

    python scripts/competition_constraints.py            # table
    python scripts/competition_constraints.py --json     # machine-readable

ponytail: the OAuth access token is refreshed by shelling out to a throwaway
`kaggle` call rather than reimplementing the refresh grant; if the CLI ever
stops rewriting ~/.kaggle/credentials.json, parse the expiry and refresh here.
"""
import json
import os
import subprocess
import sys
import urllib.request

IDS = [("arc-prize-2026-arc-agi-2", 133469),
       ("arc-prize-2026-arc-agi-3", 133468),
       ("arc-prize-2026-paper-track", 133724)]
ENDPOINT = "https://www.kaggle.com/api/i/competitions.CompetitionService/GetCompetition"
FIELDS = ["deadline", "prohibitNewEntrantsExplicitDeadline",
          "teamMergerExplicitDeadline", "kernelsPublishingDisabledDeadline",
          "maxCpuRuntimeMinutes", "maxGpuRuntimeMinutes", "maxDailySubmissions",
          "numScoredSubmissions", "leaderboardPercentage", "scoreTruncationNumDecimals",
          "requiredSubmissionFilename", "onlyAllowKernelSubmissions",
          "usesSynchronousReruns", "requiresIdentityVerification", "maxTeamSize",
          "numPrizes", "totalTeams", "totalCompetitors"]


def token():
    path = os.path.expanduser("~/.kaggle/credentials.json")
    try:
        tok = json.load(open(path))["access_token"]
        # A stale token 401s; one cheap CLI call rewrites the file with a fresh one.
        req = urllib.request.Request(
            ENDPOINT, data=json.dumps({"competitionId": IDS[0][1]}).encode(),
            headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
        return tok
    except Exception:
        subprocess.run(["kaggle", "competitions", "list", "-s", "arc-prize-2026"],
                       capture_output=True, timeout=180)
        return json.load(open(path))["access_token"]


def fetch(tok, cid):
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps({"competitionId": cid}).encode(),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json",
                 "User-Agent": "kaggle-api"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def main():
    tok = token()
    rows = {name: fetch(tok, cid) for name, cid in IDS}
    if "--json" in sys.argv:
        print(json.dumps({n: {f: d.get(f) for f in FIELDS} for n, d in rows.items()}, indent=2))
        return 0
    print(f"{'field':38}" + "".join(f"{n.replace('arc-prize-2026-', ''):>26}" for n in rows))
    for f in FIELDS:
        print(f"{f:38}" + "".join(f"{str(d.get(f)):>26}" for d in rows.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
