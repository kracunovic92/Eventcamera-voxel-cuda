# Measurement plan: event input and voxelization

## Goal and first step

Make a trustworthy baseline before moving to live streaming. The first
implementation step is a measurement contract: name each
timed boundary, record every observation with its workload, and derive summaries
after the run. Keep offline throughput and live latency as separate experiments.

The current program already writes `windows.csv`, `samples.csv`, `metadata.json`,
and `summary.json`. It measures ready-window processing, CUDA stages, and a
simulated replay schedule. The Metavision HDF5 adapter uses `Camera::from_file`
and `CameraStreamSlicer` with a one-slice queue. It does not yet measure
individual SDK decoding operations. `input_preparation_ms` combines waiting
for a slice and converting it to local events. See
[streaming experiments](streaming-experiments.md) for the existing boundaries.

### Fix the baseline before comparing paths

`apps/experiment_run.cpp` now collects stage timings in memory, stops the
`pipeline` wall clock, and then writes sample rows. Keep initialization,
capacity growth, input reading, validation, and serialization outside the
ready-window interval, and say so in metadata. Preserve separate measurements
for these excluded activities where they matter to end-to-end work.

## Metrics to record

| Boundary | Observation | Interpretation |
| --- | --- | --- |
| SDK slice | Slice wait and conversion wall time, CD count, timestamp range, queue capacity, converted window peak | The wait includes asynchronous decoding and scheduling. It is not pure decode time. |
| Window preparation | `TimeSliceSource::next()` wall time, event count, duration, empty/partial flag | Cost and size of producing each ready window. |
| CPU | CPU voxelization wall time per repetition | Ready-window processing with the specified output shape. |
| GPU transfer | H2D events, H2D grid, D2H grid: host wall time, CUDA event time, bytes | Report pageable-host-memory behavior separately from kernel work. |
| GPU compute | Clear, preprocessing, final kernel, and complete strategy: host wall and CUDA event time | The strategy interval includes clear and preprocessing. A final-kernel interval alone is not a strategy comparison. |
| Ready-window pipeline | Synchronized host wall time for each scenario | Compare only scenarios with the same required output destination. Record the destination explicitly. |
| Offline run | Total events, recording duration, input wall time, total processing wall time, events/s, windows/s | Throughput is a run-level measure; it is not a live latency number. |
| Replay model | Queue delay, deadline misses, window-end-to-result | Retain the current simulated schedule label. It excludes the actual reader queue and camera timing. |

Add correctness counters alongside timings: decoded CD events, emitted CD events,
empty windows, cap failures, invalid events, and dropped events. Expected drops
are zero for the lossless offline path. Keep an error or incomplete-run marker if
a run fails; never turn a partial run into an apparently successful summary.

For a later **live** source, also record queue depth and wait time, dropped
events/windows, and capture-to-result latency if sensor and host clocks can be
related reliably. Until that clock relationship is established, report
window-ready-to-result host latency instead. Camera timestamps and host
`steady_clock` times are different clock domains.

## Result storage

Continue using one immutable `results/run-*/` directory per invocation. Bump
the schema version when adding fields or changing timing definitions. Use:

- `metadata.json`: run ID, UTC start time, schema version, command/configuration,
  input path and hash, SDK/build/Git version and dirty state, CPU/GPU details,
  timing boundaries, host memory type, units, and completion status. Hash input
  before or after timing, not inside it.
- `input_slices.csv`: one row per SDK slice, with sequence, CD count, timestamp
  bounds, wait time, conversion time, and queue capacity. SDK-internal decode
  time is not observable through this interface.
- `windows.csv`: one row per emitted window, retaining sequence, `[start_us,
  end_us)`, event count/rate, empty and partial flags, and `source_next_ms`.
- `samples.csv`: one row per measured repetition and stage, keyed by window
  sequence, scenario, stage, and iteration. Store `wall_ms`, nullable
  `cuda_event_ms`, and byte count. No timing row is written inside its own
  measured interval. Warmups are excluded from this file.
- `summary.json`: derived from the raw rows after the run, with sample count,
  median, p95, and spread for each scenario/stage, plus total throughput and
  failure/drop counts. Keep the raw data as the authority. A p99 is useful only
  with enough independent observations; do not infer tail latency from ten
  repetitions of one window.

Use a consistent unit in every column name. An empty GPU timing field means
"not measured," whereas `0` means a measured zero. Flush or close files after
timing; a failed run should retain its raw data and an explicit failure status.
CSV and JSON are sufficient for this stage; a database would add complexity
without improving the measurement.

## How to run a comparison

1. Validate CPU/GPU output separately. Use a Release build, a fixed recording,
   fixed sensor dimensions and voxel configuration, and record all parameters.
2. Warm up each path, then measure repeated ready-window processing. Capture
   every window's event count and rate; report per-window medians as well as
   distributions across windows. Do not treat repeated runs of one window as
   independent windows.
3. Profile one representative window with NVTX/Nsight to explain a bottleneck.
   Keep profiler-instrumented durations out of baseline latency results.
4. Compare like output destinations: CPU host versus GPU host, or CPU-to-device
   versus GPU device. State whether source reading is excluded or included.
5. Run a separate sequential end-to-end pass for input throughput. Later, use
   a live camera run for queue and latency behavior instead of treating offline
   file playback as a live sensor.

## SDK integration after the baseline

The HDF5 path now uses Metavision `Camera` and `CameraStreamSlicer`. The slicer
supports fixed-duration slices with a bounded queue. Its queue limit bounds the
**number of slices**, not the events in one slice; the local event cap rejects
oversized slices after the SDK has formed them. On an SDK host, check event
counts, interval origin, empty gaps, final partial window, and event order on
the same recording before using timing results.

Official SDK references: [C++ Camera introduction](https://docs.prophesee.ai/stable/get_started/get_started_cpp.html),
[SDK Stream API](https://docs.prophesee.ai/stable/api/cpp/stream/features.html),
and [CameraStreamSlicer sample](https://docs.prophesee.ai/stable/samples/modules/stream/camera_slicer_cpp.html).

## First implementation milestone

1. Move pipeline sample writing outside the measured interval.
2. Add SDK input-slice measurements and a versioned result schema, keeping
   existing window and stage observations.
3. Check the schema and timer placement locally; on an SDK host, compare HDF5
   event/window counts with the current path.
4. Produce one short baseline report: per-stage latency by event-count band,
   input throughput, memory peaks, and any drops or cap failures.

The milestone is complete when the same raw result rows can reproduce the
summary, input and processing costs are separately visible, and no disk write
occurs inside a measured processing interval.
