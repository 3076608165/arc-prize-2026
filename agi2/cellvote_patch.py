"""The per-cell voting selection algorithm, injected into arc_decoder.py.

Kept as a separate file so the mechanism is reviewable on its own and so
`repro_nvarc.py` can splice it in without hand-editing a 38 KB notebook.

Why this shape, from measurements on the 120 public evaluation tasks:

* 32.9% of outputs land at >=90% of cells correct and score nothing, so there
  is a large near-miss population to attack.
* Upstream's `score_sum` buckets candidates by `hashable(g["solution"])`, an
  exact whole-grid hash. Two answers differing in a single cell are therefore
  treated as unrelated and compete against each other rather than reinforcing.
* Across failed outputs the two emitted attempts have error sets overlapping
  with median Jaccard 0.50 -- half of each mistake is unique to that candidate,
  which is exactly the situation a vote can decide.
* But attempt_2 alone rescues 7 of the 41 solved outputs. Any design that
  displaces it to make room for a voted grid pays 7 to win ~8, i.e. nothing.

Hence: do not add an attempt. Make the *buckets* coarser. Cluster candidates
that agree on shape and differ in at most `MERGE_TOL` of their cells, score
each cluster exactly as before, and emit a per-cell weighted vote as the
cluster's representative. Genuinely different hypotheses stay in different
clusters and still supply attempt_2; near-duplicates of one hypothesis merge
into a better version of it.
"""

CELLVOTE_SRC = '''

# --- per-cell voting selection (added by agi2/cellvote_patch.py) -------------

#: Two same-shaped candidates join the same cluster when at most this fraction
#: of their cells disagree. Near-duplicates of one hypothesis merge; genuinely
#: different hypotheses stay apart and still supply the second attempt.
MERGE_TOL = 0.10


def _disagreement(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return 1.0
    return float(np.mean(a != b))


def _weighted_cell_vote(members, weights):
    """Per-cell weighted mode over same-shaped grids."""
    grids = [np.asarray(m["solution"]) for m in members]
    out = np.array(grids[0], copy=True)
    h, w = out.shape
    for r in range(h):
        for c in range(w):
            tally = {}
            for g, wt in zip(grids, weights):
                v = int(g[r, c])
                tally[v] = tally.get(v, 0.0) + wt
            out[r, c] = max(tally.items(), key=lambda kv: kv[1])[0]
    return out


def score_cellvote(guesses, baseline=3):
    """Cluster near-duplicate candidates, vote inside each, rank as before.

    Ranking is unchanged -- clusters are scored with the same statistic
    `getter_full_probmul_3` applies to hash buckets. The only difference is
    what counts as "the same answer" and what the bucket emits.
    """
    guess_list = list(guesses.values())
    if not guess_list:
        return []

    # Heaviest candidates first, so each cluster forms around a strong member.
    def weight(g):
        return max(baseline - g["beam_score"], 1e-6)

    guess_list.sort(key=weight, reverse=True)

    clusters = []           # list of [members]
    for g in guess_list:
        for members in clusters:
            if _disagreement(members[0]["solution"], g["solution"]) <= MERGE_TOL:
                members.append(g)
                break
        else:
            clusters.append([g])

    scored = []
    for members in clusters:
        score = getter_full_probmul_3(members, baseline=baseline)
        weights = [weight(m) for m in members]
        rep = (_weighted_cell_vote(members, weights) if len(members) > 1
               else np.asarray(members[0]["solution"]))
        scored.append((score, rep))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [rep for _, rep in scored]


selection_algorithms = selection_algorithms + [score_cellvote]
'''
