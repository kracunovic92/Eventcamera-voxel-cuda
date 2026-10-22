# Research design

## Question

How do event distribution, contention, and CUDA accumulation strategy affect
the latency of temporally interpolated event-camera voxelization in fixed-time
windows?

The comparison uses a CPU reference and three CUDA strategies: event scatter,
pixel gather, and sort reduce. Block size, temporal bin count, and window
duration are controlled parameters. A slower strategy is still a useful result
when the bottleneck is measured and explained.

## Workload

The unit of work is a consecutive `[start_us, end_us)` CD-event window. It may
be empty or contain a dense burst. The SDK queues one completed slice, and the
adapter rejects a window that exceeds the configured event limit. Experiments
use original HDF5 recordings; unit tests use hand-written events.

Use representative recordings and several durations, such as 50, 100, 250,
and 500 ms. Record event counts and rates for each window rather than treating
a single count as representative of a recording. Measure CPU host, CPU grid
transfer, GPU device result, and GPU host result scenarios separately.

## Interpretation

Kernel, strategy, transfer, complete pipeline, and replay modes answer
different questions. The replay mode models a sequential schedule after each
window is ready; it does not prove live camera latency. A deployment claim also
needs an application latency budget and a defined downstream consumer. Validate
GPU output against the CPU reference before comparing speeds.
