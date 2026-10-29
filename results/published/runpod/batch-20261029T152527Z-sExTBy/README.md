# RunPod benchmark batch batch-20261029T152527Z-sExTBy

Source revision: 8abf4ac6d88bc10b37ab5d283c19d70514eeabe1

- `comparison.csv`: one row per recording, window duration, bin count, and backend.
- `stage_summary.csv`: stage timings and transferred byte counts for each run.
- `raw-results.tar.gz`: per-window `windows.csv` and `samples.csv`, plus run metadata and summaries.
- `SHA256SUMS`: checksum for the archive; verify with `sha256sum -c SHA256SUMS`.

`strategy` covers grid clear, preprocessing, and final CUDA kernel execution.
`h2d-events` and `d2h-grid` are host-to-device and device-to-host copies.
`processing-total` covers buffer preparation and the CUDA pipeline; `run_wall_ms`
also includes GPU setup, input preparation, CSV writing, and loop overhead.
`other_run_wall_ms` is the residual after measured input preparation and
processing; it does not isolate those remaining activities. Stage intervals
overlap, so do not
sum every row in `stage_summary.csv`. See `docs/streaming-experiments.md`.
