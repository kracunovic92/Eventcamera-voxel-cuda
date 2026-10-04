# Research design

## Provisional thesis question

How do event memory organization and atomic accumulation strategies affect the
latency of temporally interpolated event-camera voxelization on GPUs under
different event distributions and processing conditions?

The planned comparison is a verified CPU reference, a simple CUDA scatter
kernel, AoS and SoA event layouts, and one contention-oriented optimization
selected after profiling. Block size is a tuning parameter, not the principal
research contribution. A slower optimized variant is still a useful result if
the cause is measured and documented.

## Workload unit

The unit remains configurable until the target application is specified.
Every prepared batch carries explicit `[start_us, end_us)` bounds, event count,
sensor geometry, polarity encoding, and a source hash. The preparation tool can
select a time interval, cap the event count, or prepare a full recording. For a
count-capped prefix without an explicit end, the normalization end is the last
selected timestamp plus one microsecond. For an explicit interval, the requested
end is retained.

These workloads answer different questions:

1. Whole recordings show bulk throughput and preserve the established Sparklers
   comparison.
2. Fixed event-count selections isolate scaling with input size and are a
   practical first matrix across all seven recordings. Cover first, middle, and
   tail positions before treating the source as representative.
3. Fixed-duration windows will support a future real-time question. Their
   processing latency must be compared with the window duration and with the
   required downstream data residence. A window can contain zero events; the
   current prepared binary format rejects empty batches, so this case needs a
   format/API decision before live-stream evaluation.

No claim about drone deployment follows from batch benchmarks alone. A future
real-time experiment must specify a latency budget, event arrival behavior,
window policy, data location, and the next processing stage.

## Configurable experiment dimensions

Start with representative cases, then expand where profiling indicates a
meaningful effect. Record every selected value in the run manifest.

| Dimension | Initial examples |
|---|---|
| Event count | 10k, 100k, 500k, 1m where available |
| Temporal bins | 2, 5, 10, plus user-selected counts |
| Real data | All seven cataloged recordings |
| Synthetic distribution | Uniform, hotspot, single-pixel contention |
| Layout | AoS, SoA once CUDA exists |
| Measurement | CPU reference, GPU computation, GPU-resident, end-to-end |

The C++ test generator supports uniform and hotspot data. The benchmark
generator, `tools/generate_synthetic.py`, also supports single-pixel spatial
contention and burst timing. It saves the exact generated binary hash,
generator configuration, seed, and NumPy version. Retain the binary or verify
its hash when recreating a published synthetic case; a seed alone does not
lock the output of standard C++ distribution implementations across platforms.

## Decisions and remaining choices

The first experimental unit is a prepared 100,000-event batch. All seven raw
recordings have first, middle, and tail samples verified at 2, 5, and 10 bins.
The initial CPU measurement matrix uses the first sample from each recording,
five independent processes, ten repetitions per process, and a shuffled case
order with a recorded seed. The matrix is editable rather than hard-coded into
the benchmark executable. These CPU measurements establish a protocol and
baseline; they are not evidence of a CUDA speedup.

The CUDA study will keep the voxel definition, input batches, and measurement
boundaries fixed when comparing kernels. Start with a simple correct scatter
kernel. Profile uniform, hotspot, and single-pixel distributions; then choose
one optimization from observed bottlenecks and report its effect alongside AoS
and SoA layouts. Record both latency and numerical error for every variant.

The real-time branch still needs a chosen duration window, latency budget,
input and output residence, and target GPU. Those choices define whether a
drone-oriented claim is meaningful. A batch throughput result alone cannot
answer that question.
