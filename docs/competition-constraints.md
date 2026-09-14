# Competition constraints, read from the Kaggle API

Verified 2026-09-14 from `competitions.CompetitionService/GetCompetition`
(ids 133468 / 133469 / 133724), not from the rendered competition pages — those
are JS-only and were unreachable from this machine. Reproduce with:

```bash
python scripts/competition_constraints.py          # prints the table
python scripts/competition_constraints.py --json   # machine-readable
```

Rewritten when a value here stops matching the board. The point of this file is
that a round should not have to re-read three competition pages to check a cap.

| field | ARC-AGI-2 | ARC-AGI-3 | Paper Track |
|---|---|---|---|
| id | 133469 | 133468 | 133724 |
| final submission (UTC) | 2026-11-02 23:59 | 2026-11-02 23:59 | 2026-11-09 23:59 |
| new entrants close (UTC) | 2026-10-26 23:59 | **2026-10-26 11:59** | — |
| team merger closes (UTC) | 2026-10-26 23:59 | 2026-10-26 23:59 | 2026-11-09 23:59 |
| kernels publishing disabled (UTC) | 2026-10-26 23:59 | 2026-10-26 23:59 | — |
| runtime cap, CPU / GPU | **720 min (12 h)** | **540 min (9 h)** | — |
| submissions per day | 1 | 1 | 5 |
| **scored submissions (final count)** | **2** | **2** | **2** |
| public leaderboard share | 50% | 50% | 30% |
| score shown to | 2 decimals | 2 decimals | 5 decimals |
| required output | `submission.json` | `submission.parquet` | — |
| notebook-only (no CLI upload) | yes | yes | no |
| scored by synchronous rerun | yes | yes | — |
| **identity verification required** | **yes** | **yes** | **yes** |
| max team size | 5 | 8 | 8 |
| prizes | 8 | 5 | 3 |
| teams / competitors | 2007 / 2116 | 3034 / 3291 | — |
| entered on this account | yes | yes | yes |

Things here that were not in the repo's prior notes:

- **ARC-AGI-3 is 9 hours, not 12.** The 12-hour assumption was carried over
  from ARC-AGI-2 and is only true there.
- **Only 2 submissions ever count on both boards.** One per day, ~49 days left,
  and the score that stands is the best 2. Ranking experiments by a single
  daily result is therefore still correct, but a bad day costs nothing.
- **Identity verification is required on all three.** This is a prize gate, not
  a submission gate; unverified means submissions score and pay nothing.
- **Scored runs are synchronous reruns of the notebook.** The public number is
  not the run you watched — which is consistent with the residual ~1.7-point
  ARC-AGI-2 spread after the worker-seed fix.
- **Kernels publishing is disabled from 2026-10-26.** The notebooks must be
  public (prizes require open source) before then, not before the final deadline.
- The paper track truncates to 5 decimals and needs no notebook, but it does
  need the ARC-AGI-2 or -3 submission ID to be entered at all.

---

# Prize structure

Read 2026-09-14 from each competition's own Prizes page
(`kaggle competitions pages -c <slug> --content --page-name Prizes`) and from
`arcprize.org/competitions/2026/<track>`. This is the money, and it does not
match what the leaderboard implies.

## ARC-AGI-2 — $700,000

| pool | amount | decided by |
|---|---|---|
| Progress Prizes | $275,000 | final leaderboard rank, 8 places: 75 / 50 / 40 / 35 / 25 / 20 / 15 / 15 k |
| **Grand Prize** | **$275,000** | **the single highest-scoring Solution Writeup on six criteria** |
| Bonus Prize | $150,000 | only unlocked at ≥85% private; then split across the top 5 |

The Grand Prize is not a score prize. Its own text: "awarded to the highest
scoring Solution Writeup ... evaluated equally across the following six
criteria", each 0-5, averaged — Accuracy, Universality, Progress, Theory,
Completeness, Novelty. Artifacts must be open sourced and attached to a
Solution Writeup within seven days of the submission deadline.

## ARC-AGI-3 — $850,000

| pool | amount | decided by |
|---|---|---|
| Final leaderboard Prizes | $75,000 | rank 1-5: 40 / 15 / 10 / 5 / 5 k |
| **Milestone Prizes** | **$75,000** | **leaderboard score on a fixed date** |
| Grand Prize | $700,000 | only unlocked at 100% |

Milestone 2 is **2026-09-30**: 1st $25k, 2nd $7.5k, 3rd $5k. The only
requirements stated are a leaderboard score on the day and a notebook made
public under an open-source license by that date. Milestone 1 (2026-06-30)
has already passed.

## Paper Prize — $450,000

| pool | amount | decided by |
|---|---|---|
| Top Paper | $75,000 | 50 / 20 / 5 k across both tracks |
| **Outstanding Papers Pool** | **$375,000** | **every paper scoring above 4.5/5 — "multiple parties may win"** |

Rubric, 0-5 each: Accuracy, Universality, Progress, Theory, Completeness,
Novelty. The page is explicit that "The code submission need not achieve a
high score for the corresponding paper to be eligible", and that every paper
must link a real Kaggle submission whose score feeds only the *Accuracy* line.

## What follows from this

1. **$275K of the ARC-AGI-2 pool is a writeup prize.** Anyone reading the
   leaderboard as the only ARC-AGI-2 objective is leaving it on the table; the
   rubric is public and the criteria (Theory, Progress, Completeness) are the
   ones this project has actually been producing.
2. **ARC-AGI-3's milestone money is date-gated, not effort-gated.** Miss the
   2026-09-30 public-notebook requirement and the milestone pool is gone
   regardless of score.
3. **The paper pool has the lowest entry bar of the three.** $375K to every
   paper above 4.5, against 188 entrants, with no score requirement — versus
   $15K for 8th on a board with 2007 teams.
4. **Prizes require open source, and this repo is not licensed.** See below.

# Eligibility gap: no license, no public remote

The rules are unambiguous: "all code and methods authored by the submitter must
be made open source under a permissive public domain license (eg. CC0 or
MIT-0)", "Participants must open source their solutions before receiving
official private evaluation scores", and "participants eligible for a prize
will be removed from the competition if they do not open source their
solutions."

As of 2026-09-14 this checkout has **no LICENSE file and no git remote** — the
solutions are not public under any license, so nothing here is currently
prize-eligible. The notebooks themselves are public on Kaggle (the kernels list
API reports `isPrivate: false` for all of them), but that is not the same thing
as the code and methods being open source, and it does not cover `scripts/`,
the harness, or the analysis.

Fix, in order: add `LICENSE` (MIT-0 or CC0 for our own code, Apache-2.0
acceptable), push to a public remote, then keep pushing — the requirement is
checked at prize time, not at submission time.

