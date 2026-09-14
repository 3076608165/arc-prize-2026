# Third-party provenance

Everything authored for this project (the harness in `agi2/arclib`, `agi3/`,
`scripts/`, `experiments/`, `paper/`, `docs/`) is released under the MIT-0
licence in `LICENSE`.

Parts of the ARC-AGI-2 pipeline are derived from public Kaggle notebooks and
carry their own terms:

- `agi2/repro_nvarc.py` and the corresponding Kaggle notebook are a fork of the
  public NVARC-lineage notebook (`mikelou1/arc-agi2-lb33-89-minimal-perfpatch`)
  as described in `paper/state-of-the-art.md`. The upstream notebook's own
  stated terms apply to the derived portions; they are kept separate here so the
  distinction is visible rather than implied.
- ARC Prize data (`data/`, not distributed in this repository) is cloned from
  `github.com/arcprize`. ARC-AGI-1/2 task data is Apache-2.0; the ARC-AGI-3
  starter and game engine carry their own licence as published upstream.

`data/` is gitignored: re-create it with `scripts/setup.sh` and take the
licences from upstream, not from here.
