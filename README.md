# Event Voxel CUDA

A C++20 reference implementation and reproducible experiment framework for
temporal event-camera voxelization. CUDA implementations are the next research
phase. The current numerical rule is signed, linearly interpolated accumulation
into a `float32` tensor in `[bins, height, width]` order.

## Research direction

The proposed study measures how event memory layout, atomic contention, and
processing overhead affect CUDA voxelization latency. Whole recordings, bounded
event selections, and future fixed-duration windows are supported as distinct
workloads. The exact real-time use case remains a research choice; no GPU speedup
or real-time claim is made by this CPU-only repository state. See
[research design](docs/research-design.md).

## Reproduce the CPU pipeline

On Linux with CMake 3.24+, Ninja, a C++20 compiler, Python 3.13, Git, and Make:

```bash
make setup
make setup-external
make setup-ecf
make all
make test-python
make data-inspect-all
make data-prepare DATASET=sparklers
make verify-release DATASET=sparklers BINS=5
```

`make setup-ecf` checks out the pinned Prophesee ECF source and builds its HDF5
plugin. The external RPG comparison source is checked by `make setup-external`.
Raw files belong in `data/raw/` and are never changed by the project tools.

For a bounded sample from any recording:

```bash
make data-prepare DATASET=eye_tracking WINDOW_ID=first_100k WINDOW_MAX_EVENTS=100000
make verify-release DATASET=eye_tracking WINDOW_ID=first_100k BINS=5
```

Use `WINDOW_START_US` and `WINDOW_END_US` for explicit `[start, end)` windows.
Choose a unique `WINDOW_ID` for each selection. `make verify-all` prepares the
first 100,000 events from every raw recording and compares 2, 5, and 10-bin CPU
outputs with the pinned RPG reference and an independent float64 oracle.
`make verify-coverage` extends this to middle and tail selections from every
recording.

Controlled workloads can be generated with, for example,
`make synthetic-prepare SYNTH_CASE=hotspot_100k SYNTH_SPATIAL=hotspot`.
The generated metadata records all parameters and the binary hash.

## Benchmarks

```bash
make benchmark-smoke DATASET=sparklers BENCH_BINS=2,5,10
make benchmark-checked DATASET=sparklers BENCH_BINS=2,5,10
make benchmark-matrix
make benchmark-summary DATASET=sparklers
```

`benchmark-checked` runs both references for each measured bin count first. Full benchmark runs
require a clean Git tree; `BENCH_ALLOW_DIRTY=1` enables exploratory runs and
records the source state. Each invocation gets a unique folder containing raw
Google Benchmark JSON, a manifest, a log, and copies of validation reports.
The editable [benchmark matrix](configs/benchmark_matrix.json) covers all seven
recordings using their prepared first 100,000 events.
The older standalone CPU JSON in `results/benchmarks/` is preliminary and has no
independent-run series. See [benchmarking protocol](docs/benchmarking.md).

## Project documentation

- [Datasets and sampling](docs/datasets.md)
- [Numerical verification](docs/validation.md)
- [Reproduction and artifact layout](docs/reproducibility.md)
- [Research design](docs/research-design.md)

The build remains CPU-only until CUDA kernels and a GPU test machine are added.
