# Streaming experiments

`event_voxel_experiment` measures fixed-duration, ready-window workloads from
original HDF5 recordings.

## Build

Every build requires SDK Stream; a CPU-only build needs no GPU:

```bash
cmake --preset local-release
cmake --build --preset local-release
ctest --preset local-release --output-on-failure
```

Install Metavision SDK Stream with its ECF codec before configuring any preset.
Set `-DMetavisionSDK_DIR=<SDK-cmake-directory>` if it is outside CMake's search
path. Use `cuda-release` for CUDA on RunPod. The adapter is isolated in
`src/data/metavision/`, and the recording is read directly. On this Kali host,
OpenEB must be built from source; the prebuilt packages target Ubuntu 22.04
and 24.04. See the [official OpenEB source build guide](https://docs.prophesee.ai/stable/installation/linux_openeb.html).

Run an HDF5 CPU experiment:

```bash
./build/local-release/event_voxel_experiment \
  --input data/raw/sparklers.hdf5 --mode pipeline --scenario cpu-host \
  --window-us 100000 --bins 5 \
  --max-window-events 1000000 \
  --max-windows 100 --warmup 3 --repeat 10 --results-root results
```

The default is `--mode pipeline --scenario auto`. `mode` selects what is measured;
`scenario` selects the processing path and output destination. On a GPU host,
`auto` runs all four paths in pipeline/replay mode; on a CPU-only host it runs
`cpu-host`. For a focused four-path crossover run, use
`--mode pipeline --scenario all`.

Run the four default candidates separately with `--window-us 50000`, `100000`,
`250000`, and `500000`. `--block-size` accepts 128, 256, or 512. Strategies are
`event-scatter`, `pixel-gather`, and `sort-reduce`.

## Window and memory contract

The SDK fixes the duration and origin of time slices from its first CD events.
Intervals are consecutive, non-overlapping `[start_us,end_us)`. Empty full
slices within the recording are emitted. The final partial interval ends at
the last CD event timestamp plus one microsecond. Original event timestamps
remain in microseconds because timestamp shifting is disabled. External
triggers are excluded from voxel input.

`Camera::from_file` opens HDF5 with real-time playback and timestamp shifting
disabled. `CameraStreamSlicer` makes fixed-duration CD slices and keeps at most
one completed slice in its queue. The adapter converts each slice into reusable
`Event` storage. `EventWindowView` borrows that storage until the next source
read. A slice above `max_window_events` raises an error; this cap is checked
after the SDK creates the slice and does not bound SDK-internal memory.
`peak_window_events` measures the largest converted slice, not SDK memory.
`sizeof(Event)` is recorded to make the conversion cost explicit.

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
use schema version 2. Result folders default to ignored `results/run-*/`.

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
