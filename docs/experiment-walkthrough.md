# Experiment code walkthrough

Read `apps/experiment.cpp`, `apps/experiment_config.cpp`,
`apps/experiment_run.cpp`, `apps/experiment_metrics.cpp`, and
`apps/experiment_results.cpp` in that order.
`apps/experiment.hpp` contains the configuration, run state, and declarations.

## One invocation

1. `main` parses options, constructs `Experiment`, and turns exceptions into
   a nonzero exit code.
2. The constructor moves `Config` into `config_`, opens one HDF5 time-slice
   source, creates a unique result directory, and stores the voxel bin count.
   `source_` only opens the source; it does not read just one time slice.
3. `run` repeatedly reads a slice, records its input characteristics, and
   processes it. It stops only when `next` reports end of file.
4. After a successful full pass, `run` writes metadata and summary JSON.

`EventWindowView` borrows the adapter's converted event buffer. The next
`source_->next()` invalidates it, so all processing for a slice finishes
before another read. In a CPU build, `Experiment` owns a CPU voxelizer. In a
CUDA build, it owns one reusable `CudaExecutionSession` for the selected
strategy and a host output buffer; it does not construct or run the CPU
voxelizer.

## Functions and responsibilities

| Function | What it does and why |
| --- | --- |
| `main` | Provides the CLI error boundary. |
| `parse_integer` | Uses `std::from_chars` to require a complete valid integer and a minimum value. |
| `parse_config` | Validates the small set of common options and CUDA-only strategy and block size. |
| `Experiment::Experiment` | Initializes members in declaration order. `config_` must exist before `source_` uses its input path and duration. |
| `Experiment::run` | Owns the single pass through all CD slices, the EOF check, and final result writing. |
| `read_next_window` | Wraps one `source_->next()` call in an NVTX range in CUDA builds. |
| `record_input_window` (`experiment_metrics.cpp`) | Updates event and input-time statistics and writes one `windows.csv` row. |
| `process_window` | Runs the build's backend once for the current slice and records its phase timings after timed work. |
| `record_sample` (`experiment_metrics.cpp`) | Writes one `samples.csv` row and keeps its wall time for the summary. The CUDA-event field is optional. |
| `cpu_model` | Reads a Linux CPU model name for provenance. |
| `run_label` | Names a run from its backend and timing parameters. |
| `make_run_directory` | Creates a unique directory whose name includes the CPU reference or selected CUDA strategy. |
| `write_json` | Serializes a nlohmann JSON object with indentation and checks the output stream. |
| `describe_stage` | Calculates count, mean, nearest-rank median and p95, and spread for one phase. |
| `nearest_rank_quantile` | Sorts a copy and selects an observed nearest-rank value. For an even count, its median is the lower middle observation. |
| `ResultFiles::ResultFiles` | Opens the two CSV streams and writes their headers. |
| `write_metadata` | Builds a structured JSON object with schema, configuration, source, build, hardware, and timing definitions. |
| `write_summary` | Builds a structured JSON object with per-window statistics and whole-run throughput after EOF. |

In a CUDA build, `process_window` first reserves/reuses device buffers,
sets the current window, and resizes the host output buffer. It then times
event upload, complete strategy execution, and grid download, as well as
device-ready, pipeline, and preparation-inclusive intervals. `CudaExecutionSession` records CUDA
events on one nonblocking stream and synchronizes the stop event before each
stage returns. The CPU build times voxelization into a host grid. CSV writes
occur after the pipeline timer stops.

## NVTX and the input timer

`nvtx3::scoped_range marker("read next CD window")` starts a named range at
construction and ends it at scope exit, including exception unwinding. The
range annotates the calling CPU thread; it is not a stopwatch or a CUDA
kernel. NVTX ranges can nest, so the adapter's conversion range can appear
inside this read range. See the
[NVIDIA NVTX C++ reference](https://nvidia.github.io/NVTX/doxygen-cpp/index.html).
The application does not write NVTX ranges to its result directory. Nsight
Systems collects them into a `.nsys-rep` file when the run is profiled.

The `steady_clock` input timer starts before `read_next_window` and stops
after it returns. `input_preparation_ms` includes waiting for the SDK slice
and local conversion, plus any enabled annotation overhead. The SDK can
decode asynchronously, so this is not pure decoding time.

## Why the old controls disappeared

`mode` selected different benchmark workflows and `scenario` selected
different input/output locations. Warmup and measured repetitions repeatedly
processed the same ready window for microbenchmarks. `max_windows` stopped
before EOF; `max_window_events` rejected slices above an event count.
`profile_index` and `export_index` selected one slice for special paths.
The current experiment instead measures every time slice once, keeps the
output on the host in both builds, and computes statistics across slices.
Correctness comparisons remain in the test suite.
