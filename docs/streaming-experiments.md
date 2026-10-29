# Streaming experiments

`event_voxel_experiment` reads one HDF5 recording through Metavision SDK
Stream, divides it into fixed-duration CD-event slices, and processes every
slice once through the end of the file. There is one backend per invocation:
CPU in a CPU build, or one selected strategy in a CUDA build.

## Run

```bash
./build/local-release/event_voxel_experiment \
  --input data/raw/sparklers.hdf5 --window-us 100000 --bins 5

./build/cuda-release/event_voxel_experiment \
  --input data/raw/sparklers.hdf5 --window-us 100000 --bins 5 \
  --strategy event-scatter --block-size 256
```

`--results-root` defaults to `results`. `--window-us` defaults to 100000
microseconds and `--bins` to 5. CUDA builds accept `event-scatter`,
`pixel-gather`, or `sort-reduce` and block sizes 128, 256, or 512. Run each
strategy separately. For CPU versus GPU comparisons, run both builds on the
same machine with the same file, window duration, and bin count.

## Window and memory contract

`Camera::from_file` opens HDF5 with real-time playback and timestamp shifting
disabled. `CameraStreamSlicer` makes consecutive, non-overlapping
`[start_us,end_us)` slices and queues at most one completed slice. Empty full
slices may occur. The final partial slice ends at the last CD event timestamp
plus one microsecond. External triggers are excluded.

The adapter converts each slice into reusable `Event` storage. An
`EventWindowView` borrows that storage until the next source read, so the
experiment finishes processing before advancing. Slice duration controls
time, not event count; a dense slice can still require substantial host and
device memory. `peak_window_events` reports the largest converted slice.

## Timing and results

`results/run-*/windows.csv` records sequence, interval, CD count, event rate,
partial-final flag, and `input_preparation_ms`. That wall time covers the
call to get the next SDK slice, including waiting and local conversion. It
does not isolate SDK decoding.

`samples.csv` records `sequence,backend,stage,wall_ms,cuda_event_ms,bytes`.
The CPU build records `cpu/voxelize`, `cpu/pipeline`, and
`cpu/processing-total` as the same timed interval: voxelization into a host
grid. The CUDA build records:

- `prepare`: reserve or reuse device buffers, validate and set the window,
  and resize the host output buffer.
- `h2d-events`: copy the events to the device.
- `strategy`: clear output, preprocess if needed, and run the final kernel.
- `device-ready`: wall time from H2D start until the grid is ready on the
  device, excluding buffer preparation and D2H.
- `d2h-grid`: copy the grid to host memory.
- `pipeline`: synchronized host wall time covering H2D, strategy, and D2H.
- `processing-total`: wall time for buffer preparation plus the pipeline.

The pipeline timer excludes input preparation, buffer preparation, and CSV
writes. CUDA stage rows contain both synchronized CUDA event and host
wall time. Host memory is pageable. The `cuda_event_ms` field is blank where
it was not measured. The grid is computed for each window but not persisted.

`summary.json` reports count, mean, nearest-rank median and p95, and
population standard deviation for each stage across all slices, plus event
counts, input-preparation statistics, and whole-run throughput. `metadata.json`
records configuration, source, build, CPU and GPU details, and timing definitions.
Run folders include backend and key parameters, such as
`run-cpu-reference-w100000us-b5-*` or
`run-cuda-event-scatter-w100000us-b5-t256-*`. The result schema is version 4.
A run that fails before EOF does not write
metadata or summary; existing CSV rows remain in its run directory.

Offline HDF5 playback measures ready-window processing. It does not by itself
measure live camera capture-to-result latency or a sustained live queue.

## Comparing CPU and CUDA

Run both builds on the same machine with the same recording, slice duration,
bin count, and sensor resolution. Join `windows.csv` and `samples.csv` by
`sequence`; inspect event count and rate beside each time. For a host-grid
consumer, compare `cpu/processing-total` with
`<strategy>/processing-total`. CUDA `device-ready` shows when its grid is
ready for a GPU consumer, but this experiment does not measure the competing
CPU-voxelization-plus-grid-upload path. It cannot establish that device-output
break-even point from these two runs alone.

Stage timings show where time goes: event H2D, strategy execution, grid D2H,
and preparation. `run_wall_ms` and its throughput rates include input
waiting and CSV writing, but exclude source opening and JSON output. Each
CUDA stage currently synchronizes its stop event, so the measured pipeline
reflects sequential stages rather than an overlapped streaming design.

## Profiling

The `cuda-profile` build adds source information for Nsight. These wrappers
profile the whole recording, so use a short recording for focused traces:

```bash
scripts/profile_nsys.sh results/profiles/event-scatter-timeline data/raw/sparklers.hdf5 \
  --strategy event-scatter
scripts/profile_ncu.sh results/profiles/kernels data/raw/sparklers.hdf5 \
  --strategy event-scatter
```

CUDA builds require NVTX3 headers. NVTX ranges label input reads, conversion,
transfers, output clearing, preprocessing, and final kernels. Profiler timing
should be kept separate from baseline latency results. The experiment's
`windows.csv` and `samples.csv` contain its own clock measurements; NVTX
ranges are collected only when the program runs under Nsight Systems. The
Nsight Systems wrapper writes `results/profiles/event-scatter-timeline.nsys-rep`
for the example above. Open that report in Nsight Systems to see the named
ranges beside CUDA activity, or run
`nsys stats --report nvtx_sum results/profiles/event-scatter-timeline.nsys-rep`
for an aggregate range summary.
