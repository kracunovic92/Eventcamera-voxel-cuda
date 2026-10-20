# Numerical validation

For an event `(x, y, t, p)` in `[t0, t1)`, polarity `p=1` contributes `+1` and
`p=0` contributes `-1`. Its continuous temporal position is
`u=(B-1)(t-t0)/(t1-t0)`. Bin `b` receives the signed triangular weight
`max(0, 1-|b-u|)`. The output is C-contiguous `float32` in `[B,H,W]` order.

The C++ reference calculates temporal position in `double`, converts each
weight to `float`, and accumulates in `float`. CUDA may reorder additions when
multiple threads update one voxel, so voxel-by-voxel numerical comparison and
error distributions matter more than bitwise equality.

## Checks in increasing scope

1. Hand-calculated Catch2 cases cover start, end-near, exact and fractional bin
   positions, polarity, independent pixels, nonzero time origins, collisions,
   invalid input, and spatial conservation.
2. Python tests compare the complete tensor with an independently coded
   triangular-weight oracle under uniform, hotspot, and single-pixel events for
   several bin counts.
3. A tiny HDF5 round trip checks inspection, binary preparation, C++ loading,
   NumPy export, selection metadata, and malformed empty-window behavior.
4. The adapted RPG E2Depth comparison checks every voxel of selected real
   recordings. The script requires the pinned Git revision and source SHA-256,
   and rejects local modifications to the reference source.
5. A separate float64 triangular-weight accumulation checks every voxel against
   the mathematical rule without using RPG's accumulation code. It processes
   events in bounded chunks and records error statistics and the worst voxel.

`make verify-all` applies checks 4 and 5 to the first 100,000 events of every raw
recording for 2, 5, and 10 bins by default. `validate_outputs.py single` writes
one `bins_N.validation.json` report containing both references for each
recording, window, and bin count. `make benchmark-checked` repeats both checks
for every bin count it is about to measure. The generated summary at
`results/reports/validation_first_100000.json` ties these reports
to all raw source fingerprints and prepared binary hashes.
`make verify-coverage` additionally checks middle and last 100,000-event
selections from each recording, using source event indices to avoid treating an
early prefix as representative of the whole file. Each position gets its own
summary. Full-recording voxel validation remains a separate, memory-intensive
experiment.

Both comparisons currently use absolute tolerance `1e-3` and relative tolerance
`1e-4` per voxel. The relative term matters for highly contended voxels, so
include single-pixel collision workloads in numerical validation.
`validate_outputs.py capture` applies the same two references to every saved CPU
and CUDA grid in a RunPod session, writing `validation.csv`. Before accepting CUDA results,
characterize error on collision workloads and record failed voxel count,
maximum absolute error, mean absolute error, RMSE, and the worst voxel.

## One validation command for saved experiments

`scripts/experiments/validate_outputs.py` has three modes:

| Mode | Input | Output | Use |
|---|---|---|---|
| `single` | One prepared `events.bin` and saved `.npy` grid | Combined JSON report | Check an individual CPU or CUDA output. |
| `capture` | A complete capture directory and its prepared dataset root | `validation.csv` | Check every configured CPU and CUDA output against both independent references. This is the pipeline gate. |
| `compare-backends` | A complete capture directory | `comparisons.csv` | Investigate differences between saved CPU and CUDA outputs after the reference check. |

For example:

```bash
python3 scripts/experiments/validate_outputs.py single \
  --events <prepared-events.bin> --voxels <saved-grid.npy> \
  --report <validation.json> --backend cuda --strategy event-scatter
python3 scripts/experiments/validate_outputs.py capture \
  <capture-directory> --dataset-root <prepared-dataset-root>
python3 scripts/experiments/validate_outputs.py compare-backends \
  <capture-directory>
```

`single` and `capture` check both the pinned RPG implementation and the float64
oracle by default. `--references precision` can select the float64 check for an
exploratory run without the RPG checkout. The two reference implementations live
in `reference_models.py` as internal code; there is one validation entry point.
