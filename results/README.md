# Results

- `metadata/`: source inspection reports for all recordings; keep these in Git.
- `validation/`: RPG and independent float64 comparison reports; keep these in Git.
- `benchmarks/<dataset>/<window>/<run-id>/`: raw benchmark JSON, manifest,
  log, and copied validation reports; keep research runs in Git or an immutable
  archive. The manifest records hashes, matrix position, and whether the source
  tree was dirty. Result files created during a matrix run do not change the
  recorded source state.
- `reports/`: summaries across independent benchmark processes; keep in Git.
- `voxels/` and `visualizations/`: generated arrays and figures, ignored by Git.
  Copy selected publication figures to `docs/figures/` when needed.

The existing `cpu_20261002_214143.json` is a preliminary single-process CPU
benchmark with ten repetitions per case. It predates the structured manifest
layout and should remain labeled as preliminary.
