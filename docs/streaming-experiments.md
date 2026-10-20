# Streaming experiments

`event_voxel_experiment` is the fixed-duration, ready-window experiment path.
The older `run_cpu`, `run_cuda`, Google Benchmark, and Python `.bin` preparation
workflows remain available for historical comparisons. They are **legacy
whole-batch experiments**, not faithful event-rate replay; `sample_evenly`
changes density and is never used by this executable.

## Build

CPU-only synthetic smoke tests need no SDK or GPU:

```bash
cmake --preset local-release
cmake --build --preset local-release
ctest --preset local-release --output-on-failure
./build/local-release/event_voxel_experiment --source synthetic --mode pipeline \
  --scenario cpu-host --synthetic-events 100000 --window-us 50000
```

For direct `.hdf5`, install Metavision SDK Stream with its ECF codec and configure
`cmake --preset hdf5-cpu-release -DMetavisionSDK_DIR=<SDK-cmake-directory>`.
For CUDA on RunPod use `cmake --preset cuda-hdf5-release
-DMetavisionSDK_DIR=<SDK-cmake-directory>`, then build that preset. The
`EVENT_VOXEL_ENABLE_METAVISION` option is explicit; configuring it without an
SDK fails with CMake's package diagnostic. The adapter is isolated in
`src/data/metavision/`. No Python converter or `.bin` is used by the new path.
This checkout has no installed SDK, so the adapter still needs a build and a
real-recording integration run on an SDK host.
The current [RunPod image](../infra/runpod/Dockerfile) supports the older
prepared `.bin` pipeline; add OpenEB SDK Stream and its ECF codec to a
versioned image before running this direct HDF5 path on a pod.

Run an HDF5 CPU experiment:

```bash
./build/hdf5-cpu-release/event_voxel_experiment --source hdf5 \
  --input data/raw/sparklers.hdf5 --mode pipeline --scenario cpu-host \
  --window-us 100000 --bins 5 --chunk-us 10000000 \
  --max-chunk-events 1000000 --max-window-events 1000000 \
  --max-windows 100 --warmup 3 --repeat 10 --results-root results
```

For a four-path crossover run on a GPU host, use `--mode pipeline --scenario all`.
Run the four default candidates separately with `--window-us 50000`, `100000`,
`250000`, and `500000`. `--block-size` accepts 128, 256, or 512. Strategies are
`event-scatter`, `pixel-gather`, and `sort-reduce`.

## Window and memory contract

The first CD event timestamp is the interval origin. Intervals are consecutive,
non-overlapping `[start_us,end_us)`. Empty intervals within the recording are
emitted. The final interval is emitted through the last CD event timestamp plus
one microsecond and marked `partial_final`. Original event timestamps remain in
microseconds; the HDF5 index offset is not applied to decoded CD timestamps.
Only the SDK CD callback is registered; external triggers are excluded.

`EventWindowView` borrows chunk memory. It is valid until the next
`EventWindowSource::next()` call. A window contained within one chunk is a span
into that chunk; a boundary-spanning window is copied once into reusable
staging memory. The chunk stops after `chunk_us` or `max_chunk_events`. A window
above `max_window_events` raises an error rather than dropping data. The SDK
decoded callback queue uses the chunk event cap as its own cap. Thus live input
memory is bounded by the decoded queue, chunk, staging window, and one lookahead
event. `peak_buffered_events` reports the source's chunk/staging/lookahead peak;
it does not include the SDK queue. `sizeof(Event)` is recorded because it is
usually 24 bytes, whereas the old disk record was 16 bytes.

## Measurement boundaries

- `kernel`: ready device input, required gather/sort preprocessing and output
  clear occur before the CUDA event interval. The timed stage is the final
  strategy kernel. This is not a complete strategy comparison.
- `strategy`: output clear, grouping or sorting and reduction, and the final
  kernel. Input transfer, host preparation, and allocations are outside.
  Common and strategy scratch buffers are reserved beforehand; Thrust's
  internal scan/sort/reduce implementation may still allocate during a
  strategy sample, and this is marked in metadata.
- `transfer`: three independent H2D events, H2D CPU voxels, and D2H grid
  samples with byte counts. Host memory is pageable, transfers use one explicit
  CUDA stream, and both synchronized CUDA event and host wall times are saved.
- `pipeline`: CPU host, CPU then grid H2D, event H2D then GPU device result,
  and GPU with grid D2H. `pipeline` samples are synchronized host wall times;
  stage rows are also saved. Grid allocation and CPU output zeroing are part
  of CPU processing. CUDA input/output capacity is reserved before timed runs.
- `replay`: repeats the selected ready-window pipeline, then simulates a
  sequential schedule using each window's median processing time and release
  at the **end** of its interval. It reports queue delay, deadline misses,
  idle time, and time from window end to result. The deadline is one configured
  window duration after release. Input preparation time and windows where it
  exceeded an interval are reported separately; this does not model a live
  reader queue or prove camera latency.
- `profile`: one selected window with a single repetition and NVTX ranges
  when NVTX3 headers are installed. Select it with `--profile-index N` and set
  `--max-windows` above N. Use the Nsight wrappers below. Profiled
  durations are not baseline latency samples.
- `validate` and `export`: separate, unmeasured invocations. `validate`
  compares all three GPU strategies to the CPU reference where a GPU exists;
  on a CPU build it checks the CPU path. `export` writes one selected CPU `.npy`
  grid and, where available, the selected GPU strategy grid in `outputs/`,
  selected by `--export-index`.

`windows.csv` records each interval, event rate, final flag, and combined input
decode/window preparation wall time. `samples.csv` has every measured stage
sample, scenario, iteration and byte count. `crossovers.csv` gives each window's
CPU-to-device and GPU-device medians where both are measured. `metadata.json`
records configuration, source, CPU and available GPU details, build and Git
commit, capacities, event size, and inclusion flags. `summary.json` provides
descriptive statistics, event counts, crossover counts and replay data. These
use schema version 1. Result folders default to ignored `results/run-*/`.

The crossover compares synchronized host wall time for the same ready window.
`cpu-host`, `cpu-to-device`, `gpu-device`, and `gpu-to-host` have different
output destinations and should be read as such. There is no neural network.
The oldest event is about one window duration old when the interval releases;
oldest-event-to-result age also includes backlog and ready-window processing.
No result here implies downstream detection accuracy.

## Profiling

On a GPU host with SDK and Nsight installed:

```bash
scripts/profile_nsys.sh results/profiles/timeline data/raw/sparklers.hdf5
scripts/profile_ncu.sh results/profiles/kernels data/raw/sparklers.hdf5
```

Use `cuda-profile` (`RelWithDebInfo`, `-lineinfo`) for source correlation and
Release for latency. Do not use CUDA `-G` for benchmarking. `ncu --set full`
requests occupancy, memory, stalls, and atomic metrics; some counters need
hardware permissions. Nsight wrappers print an actionable message when the
tool is absent. `nvtx3` annotations cover input preparation, transfers, clear,
preprocessing and final kernels when its header is found.

## Migration note

The old `.bin` readers, conversion scripts, one-shot JSON writer and their
tests/docs remain because the existing validation pipeline and research
artifacts still call them. The new executable never calls them. Remove them
only after their callers and historically valuable benchmark data are migrated.
Raw `.hdf5` recordings and local generated data are untouched.
