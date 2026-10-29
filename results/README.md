# Results

`event_voxel_experiment` creates an ignored `run-*/` folder for each
invocation. It contains `windows.csv`, `samples.csv`, `metadata.json`, and
`summary.json`. RunPod downloads live in ignored `results/runpod/`.
The per-window timings and stage definitions are described in
[streaming experiments](../docs/streaming-experiments.md).

Published batches live in tracked `results/published/runpod/`. Each includes
comparison tables and a compressed archive of the original CSV/JSON files,
with a SHA-256 checksum. The [RunPod guide](../docs/runpod.md) explains how to
download, inspect, and publish a batch.
