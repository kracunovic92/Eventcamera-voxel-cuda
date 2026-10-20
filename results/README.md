# Results

Generated results stay on the machine that produced them and are ignored by
Git. Historical result files that were once tracked remain available in Git
history; existing local copies are left in place during this cleanup.

| Folder | Created by |
|---|---|
| `metadata/`, `validation/`, `reports/` | Dataset inspection, reference checks, and summary commands. |
| `benchmarks/`, `experiments/` | Legacy CPU and RunPod benchmark pipelines. |
| `run-*/` | Fixed-duration `event_voxel_experiment` runs. |
| `downloads/` | Verified archives fetched from RunPod. |

Keep the archive and its `.sha256` file before ending a RunPod pod. See the
[RunPod runbook](../docs/runpod.md) for local download and optional Drive copy.
