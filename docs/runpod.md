# RunPod CUDA benchmark runbook

## Scope

The first GPU session validates the CUDA build, strategy correctness, smoke
matrix, and raw result capture. The current benchmark reports host end-to-end
latency for the cold `CudaVoxelizer::voxelize()` API. It includes allocation,
host-to-device transfer, computation, device-to-host transfer, and cleanup.
CUDA phase timing and GPU-resident or buffer-reuse modes are future work.

The committed matrices use deterministic `uniform` and `collision-heavy`
synthetic inputs. A real-camera matrix requires a fixed, versioned input file
and is not included yet. Do not describe these runs as real-data results.

## Prepare a known revision

Build and publish `infra/runpod/Dockerfile` to a registry accessible to your
RunPod account. The image uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` and
installs CMake, Ninja, a C++ compiler, Python 3, Git, and certificates.
Record the published image digest with the experiment. A RunPod GPU pod needs
enough storage for the image, build tree, and results. Put the checkout at
`/workspace/event-voxel-cuda` and retain results at
`/workspace/experiment-results` (or set `EVENT_VOXEL_RESULT_ROOT`).

Commit and push the desired source revision before creating the pod. In the
pod, clone the repository and check out that exact commit. The runner rejects
a dirty checkout for official runs. The pod needs network access to fetch the
pinned Catch2 and Google Benchmark source dependencies during configuration.

```bash
cd /workspace
git clone <repository-url> event-voxel-cuda
cd event-voxel-cuda
git checkout <commit-sha>
./scripts/runpod/run_all.sh --smoke-only
```

`run_all.sh` checks the environment and working tree, builds the Release CUDA
preset, runs CTest and the `[cuda]` correctness suite, then runs the smoke and
full matrices in that order. Use `--smoke-only` for the inexpensive first pod
validation; run without it for the full matrix after reviewing smoke output.
Any failed step stops the script. CUDA tests skip
without a device during local compilation; `run_all.sh` additionally runs
`event_voxel_run_cuda`, which fails if no CUDA device is usable.

Each matrix creates a new directory under `smoke/` or `full/` with the exact
config, raw `measurements.csv`, `summary.csv`, Git and host metadata, GPU query,
environment output, and correctness output. The directory name includes UTC
time, GPU name, commit prefix, and a random suffix. A failed matrix keeps its
partial CSV and marks its metadata as failed. Do not use a failed run for
performance claims.

## Preserve results before ending the pod

After the script succeeds, archive `/workspace/experiment-results`, transfer
the archive to local storage, then inspect the local archive and CSV files.
Only then terminate the paid pod. For example:

```bash
tar -C /workspace -czf /workspace/event-voxel-results.tar.gz experiment-results
tar -tzf /workspace/event-voxel-results.tar.gz | head
```

Transfer the archive with your preferred RunPod or SSH file transfer method.
The project does not upload results automatically.

## Local verification

The CPU test preset and CUDA compile preset can be validated without a GPU.
Actual CUDA numerical behavior, benchmark latency, and the container image
must be verified on a GPU pod. The first pod session should be treated as a
pipeline validation run before relying on full thesis measurements.
