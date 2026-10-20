# Benchmarking protocol

For the fixed-duration HDF5 and ready-window benchmark path, use the
[streaming experiment protocol](streaming-experiments.md). The remainder of
this document describes the preserved legacy `.bin` workflows.

## Current CPU measurement

`event_voxel_bench_cpu` loads a prepared binary before timing. Every timed
iteration calls `CpuReferenceVoxelizer::voxelize()`, including its event
validation pass, output allocation and zeroing, interpolation, and accumulation.
File I/O, NumPy export, and external correctness comparison are excluded.
The benchmark uses wall time and reports events per second. Bin counts come
from `EVENT_VOXEL_BENCH_BINS` or `BENCH_BINS` through Make.

`make benchmark-checked` verifies all measured bin counts, then starts a
benchmark process. A run folder contains `benchmark.json`, `benchmark.log`,
`manifest.json`, and a copy of the combined RPG and float64 validation report. The
manifest records source state and file hashes, input and output parameters,
build flags, system information, and the exact command. Clean source is required for a formal run;
use `BENCH_ALLOW_DIRTY=1` only for exploratory work.

`configs/benchmark_matrix.json` declares seven prepared recording samples,
their bins, repeat count, warmup, minimum measurement time, validation gate,
and a fixed seed for shuffling case order each round. Run it with
`make benchmark-matrix` after `make verify-all`. Change the JSON or provide
`MATRIX_CONFIG=...` to select another experiment. Each child run manifest stores
the matrix file hash, case, round, and execution order. For a short pipeline
check, use `MATRIX_CONFIG=configs/benchmark_smoke.json MATRIX_ALLOW_DIRTY=1`;
its short duration does not support performance claims.
`make verify-coverage` can supply verified middle and tail selections for a
broader matrix. Add those dataset/window pairs to a copy of the JSON when
comparing source position effects.

Run multiple independent processes under stable power, cooling, and background
load. `make benchmark-summary` reports the median of each process's repetition
medians and does not pool individual Google Benchmark iterations across
processes. The summary groups exact source snapshots, executables, datasets,
and measurement settings; it excludes dirty exploratory runs unless
`--include-dirty` is specified. The preliminary 2026-10-02 CPU JSON predates
this protocol.

## Current CUDA measurement

`event_voxel_bench_cuda` loads and selects events before timing. It warms up the
selected CUDA strategy, then times repeated calls to
`CudaVoxelizer::voxelize()` with `std::chrono::steady_clock`. The call allocates
device buffers, copies the events to the device, clears the output, runs the
strategy, synchronizes, copies the grid back, and releases device buffers. A
host clock is appropriate for this complete, host-observed call because the
call cannot return before GPU work and the output copy finish. Its CSV field is
`end_to_end_ms`; it does not measure kernel-only device execution.

`scripts/experiments/run_matrix.py` reads the CUDA JSON matrix, verifies input
hashes, calls the C++ executable for each case, and retains every measurement
with hardware and source metadata. `summarize_results.py` derives descriptive
statistics from that CSV. The `run_cpu` and `run_cuda` apps separately save
voxel arrays. `validate_outputs.py capture` compares each saved output directly
with the pinned RPG implementation and the float64 oracle before performance
measurements begin. The
apps' `voxelize_ms` fields use a host clock around one call and are diagnostic
rather than the repeated performance measurements used for summaries.

## Planned GPU measurements

Record each boundary separately:

| Name | Included work |
|---|---|
| GPU computation | Output clear plus voxelization kernel, device-resident input |
| GPU kernel | Kernel alone with a correctly reset output outside timing |
| GPU-resident pipeline | Host-to-device input plus GPU computation, output remains on GPU |
| Host end-to-end | Allocation policy, transfers, clear, kernel, output copy, synchronization |

Use CUDA events on the same stream for device execution and a synchronized
host timer for complete host-observed latency. Record warmup, repetitions,
device and driver versions, clocks/power state, event layout, block size, and
numerical error. A future fixed-duration-window experiment should compare
processing latency with its declared window budget. GPU speedups are not yet
reported.
