# Event Voxel CUDA

A C++20 reference implementation and reproducible experiment framework for
temporal event-camera voxelization. The repository includes three CUDA baseline
strategies. The current numerical rule is signed, linearly interpolated accumulation
into a `float32` tensor in `[bins, height, width]` order.

## Research direction

For new fixed-time, bounded HDF5 experiments use the
[streaming experiment guide](docs/streaming-experiments.md). The `event_voxel_experiment`
CLI reads `.hdf5` directly through an optional Metavision C++ adapter and
records separate CPU, CUDA strategy, transfer and ready-window timings. The
workflow below is retained for historical whole-batch comparisons.
The [project map](docs/architecture.md) shows how the new and historical paths connect.

The proposed study measures how event memory layout, atomic contention, and
processing overhead affect CUDA voxelization latency. Whole recordings, bounded
event selections, and fixed-duration windows are supported as distinct
workloads. The exact real-time use case remains a research choice; no GPU speedup
or real-time claim is made before measured RunPod results. See
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

The following are legacy whole-batch experiments. For the current streaming
measurement boundaries and artifact schema, see the
[streaming experiment guide](docs/streaming-experiments.md).

The two similarly named code areas have different jobs:

| Location | Purpose |
|---|---|
| `apps/run_cpu.cpp`, `apps/run_cuda.cpp` | Run one case, save its voxel grid and a diagnostic JSON record. |
| `benchmarks/bench_cpu.cpp`, `benchmarks/bench_cuda.cpp` | Measure repeated voxelization calls in C++. |
| [`scripts/experiments/`](scripts/experiments/README.md) | Run the local or RunPod pipeline, capture saved grids, check them against RPG, and summarize measurements. |
| `scripts/runpod/` | Prepare the pod and execute the complete remote workflow. |

For CUDA, the timed C++ call synchronizes the GPU and returns the output to
host memory. The reported latency is the full host-observed call, including
GPU work and transfers; kernel-only GPU timing is not yet implemented. See
[benchmarking protocol](docs/benchmarking.md) for the exact timing boundary.

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
See the [benchmarking protocol](docs/benchmarking.md) for the required
independent-run series and validation gate. Generated results stay outside Git.

## Project documentation

- [Datasets and sampling](docs/datasets.md)
- [Numerical verification](docs/validation.md)
- [Reproduction and artifact layout](docs/reproducibility.md)
- [Python script map and pipeline stages](docs/python-scripts.md)
- [Research design](docs/research-design.md)

## CUDA build and RunPod preparation

The CUDA backend is optional. For a local compile check, activate the `cuda-dev`
Conda environment, which supplies `nvcc`, then run:

```bash
cmake --preset cuda-local-check
cmake --build --preset cuda-local-check
ctest --preset cuda-local-check --output-on-failure
```

The local preset uses the Conda CUDA toolkit and `/usr/bin/g++-15`. CUDA tests
skip when no NVIDIA GPU is available, so the compile check alone does not
establish GPU correctness. The `cuda-release` preset is for the remote GPU
container. The committed RunPod matrices require the prepared real recordings;
they are packaged separately from Git. See [the RunPod runbook](docs/runpod.md)
for dataset transfer, image, execution, result transfer, and validation steps.

Run `make pipeline-local SMOKE=1` on the local machine to prepare a small,
externally verified dataset archive. Transfer that archive to the pod and run
`make pipeline-runpod SMOKE=1` there. Remove `SMOKE=1` for full runs after the
smoke workflow succeeds. See [the experiment script map](scripts/experiments/README.md).

`event_voxel_run_cpu` and `event_voxel_run_cuda` save individual voxel grids and
JSON run records. The CUDA app requires `--strategy event-scatter`,
`--strategy pixel-gather`, or `--strategy sort-reduce`. The RunPod driver captures
all configured CPU and CUDA grids under `data/outputs/` in its session archive,
then compares each grid directly with the pinned RPG and float64 references using
`scripts/experiments/validate_outputs.py capture`. Repeated
performance measurements remain in the separate benchmark executables.
