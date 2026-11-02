"""Cover figure: identical-code draw distributions for our two final builds.

Data are the leaderboard scores of byte-identical resubmissions, read from
Kaggle on 2026-11-02 (see paper/findings.md). Run: python paper/assets/make_draws_figure.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ponytail: hard-coded draw lists, not a Kaggle API pull -- this is a figure,
# the numbers are already checked into findings.md with their refs.
AGI2 = [29.31, 30.56, 30.97, 28.89, 29.31, 30.14, 29.72, 30.28, 30.28,
        28.06, 28.47, 28.89, 29.31, 28.47]
AGI3 = [28.12, 26.03, 20.98, 27.96]

fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200)
for a, vals, name, pair, med in [
    (ax[0], AGI2, "ARC-AGI-2: 14 draws of one pinned build", (30.97, 30.56), 29.31),
    (ax[1], AGI3, "ARC-AGI-3: 4 draws of one byte-identical agent", (28.12, 27.96), 2.50),
]:
    mean = sum(vals) / len(vals)
    sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    a.axhspan(mean - sd, mean + sd, color="#4c72b0", alpha=0.15, lw=0,
              label=f"mean {mean:.2f} ± {sd:.2f}")
    for i, v in enumerate(sorted(vals)):
        a.plot(i, v, "o", ms=7, color="#1f3b73")
    a.axhline(mean, color="#4c72b0", lw=1)
    a.axhline(pair[0], color="#c44e52", lw=1.2, ls="--",
              label=f"selected pair {pair[0]:.2f} / {pair[1]:.2f}")
    if name.startswith("ARC-AGI-3"):
        a.axhline(med, color="grey", lw=1, ls=":",
                  label=f"field median {med:.2f}  (max 75.56)")
    a.set_title(name, fontsize=9)
    a.set_ylabel("public leaderboard score", fontsize=8)
    a.set_xlabel("draw index (identical bytes)", fontsize=8)
    a.tick_params(labelsize=7)
    a.legend(fontsize=6.5, loc="lower right")
    a.margins(x=0.08)

fig.suptitle("One submission is not one measurement", fontsize=10)
fig.tight_layout()
out = "paper/assets/draw-distributions.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

# check: the spreads the paper quotes
assert abs((max(AGI2) - min(AGI2)) - 2.91) < 0.005
assert abs((max(AGI3) - min(AGI3)) - 7.14) < 0.005
print("ranges ok: %.2f and %.2f" % (max(AGI2) - min(AGI2), max(AGI3) - min(AGI3)))
