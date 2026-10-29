# Measurement plan

## Current baseline

One invocation processes every fixed-duration slice in one HDF5 recording
once. A CPU build uses the CPU voxelizer; a CUDA build uses one selected CUDA
strategy. Run the builds and strategies separately with the same recording,
duration, bin count, and machine. See
[streaming experiments](streaming-experiments.md) for exact timing boundaries
and result columns.

Each `windows.csv` row is an observation of the workload: time interval,
event count, event rate, and input preparation time. Each `samples.csv` row
is one phase timing for that window. `summary.json` describes distributions
across windows. This avoids treating repeated executions of one window as
independent input samples.

## Interpretation

- `input_preparation_ms` measures the call that waits for the SDK slice and
  converts events into local storage. It does not isolate HDF5 decoding.
- CPU `pipeline` is voxelization into a host grid. CUDA `pipeline` is event
  upload, strategy execution, and grid download into a host buffer. Both
  exclude the source read and CSV writes.
- CUDA `prepare` measures buffer reserve/reuse, window setup, and host output
  sizing separately. Capacity growth may make early windows slower.
- CUDA event time and host wall time answer different questions. Stage calls
  synchronize their stop event; the enclosing pipeline is host wall time.
- Offline HDF5 playback cannot establish camera capture-to-result latency or
  a live queue deadline. A live source will need separate clock and queue
  measurements.

## Before a performance claim

1. Run numerical tests, including CPU/CUDA agreement, on the target GPU.
2. Run the HDF5 integration test on an actual recording.
3. Measure each backend on the same host with matching input and voxel shape.
4. Inspect per-window event counts alongside phase timings, especially dense
   and empty windows. Report sample counts and the raw CSV data.
5. Use Nsight on a short recording to explain a bottleneck. Keep profiler
   durations separate from baseline timings.

The SDK slicer queues one completed slice, but slice duration does not cap
the events or memory in that slice. Record the largest slice and peak device
buffer capacities. Save an input hash and exact build environment with the
result folder.

## Follow-up instrumentation

The current schema records converted-window peaks and phase timings. Future
measurements could add an explicit run status on failure, exact SDK decoding
time if its API exposes it, and total run throughput. Live capture would
require queue depth, dropped events/windows, and a reliable mapping between
camera timestamps and host clock time.
