# Results

`event_voxel_experiment` creates a new ignored `run-*/` folder for each
invocation. It contains `windows.csv`, `samples.csv`, `crossovers.csv`,
`metadata.json`, and `summary.json`; export mode can also save selected `.npy`
grids. These generated files are not tracked by Git.

Archive the run folder, its SHA-256, and the source recording hash before
ending a temporary RunPod pod. The [RunPod guide](../docs/runpod.md) explains
network-volume retrieval. Historical measurements remain in Git history; local
copies of earlier results are not part of the current workflow.
