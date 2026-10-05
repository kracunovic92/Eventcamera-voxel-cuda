# RunPod CUDA benchmark runbook

## Scope

The first GPU session validates the CUDA build, strategy correctness, smoke
matrix, and raw result capture. The current benchmark reports host end-to-end
latency for the cold `CudaVoxelizer::voxelize()` API. It includes allocation,
host-to-device transfer, computation, device-to-host transfer, and cleanup.
CUDA phase timing and GPU-resident or buffer-reuse modes are future work.

The committed smoke and full matrices use prepared real-camera recordings.
Each input path and SHA-256 hash is fixed in the configs. The benchmark loads
the binary outside the timed region and selects evenly spaced, ordered events
across the recorded window for each event count. All strategies receive the
same selected events. Raw HDF5 recordings stay on the local machine.

## Transfer the real-data inputs

The prepared binaries are ignored by Git. Package the eight exact inputs
named by the two configs on the local machine:

```bash
python3 scripts/benchmark/package_datasets.py \
  --config configs/cuda_benchmark_smoke.json \
  --config configs/cuda_benchmark_matrix.json \
  --output data/exports/runpod-real-data.tar.gz
```

The packager verifies every binary against its committed hash and its
`metadata.json`, then prints the archive hash. Keep the original HDF5 files
and this archive in local storage. Transfer the archive to the pod using the
pod's SSH address and port, for example:

```bash
scp -i <private-key> -P <ssh-port> data/exports/runpod-real-data.tar.gz \
  root@<pod-host>:/workspace/event-voxel-real-data.tar.gz
```

Alternatively, put the archive at an HTTPS URL accessible from the pod. Set
`EVENT_VOXEL_DATA_URL` to that URL. The run script downloads the archive,
extracts it outside Git, and verifies every binary hash before building or
benchmarking. A persistent network volume can hold the extracted dataset root
for later pods; `EVENT_VOXEL_DATA_ROOT` changes its location.

```bash
EVENT_VOXEL_DATA_URL='<private-https-url>' ./scripts/runpod/run_all.sh --smoke-only
```

## Prepare a known revision

Build and publish `infra/runpod/Dockerfile` to a registry accessible to your
RunPod account, using the repository root as Docker build context. The image
uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` and installs CMake, Ninja, a C++
compiler, Python 3, Git, certificates, and OpenSSH. Add your public SSH key to
your RunPod account and expose TCP port 22 when creating the pod. The startup
script receives `PUBLIC_KEY`, starts SSH, and enables SCP transfer.
For example, on a machine with Docker access:

```bash
docker build -f infra/runpod/Dockerfile -t <registry>/<image>:cuda12.8.1 .
docker push <registry>/<image>:cuda12.8.1
```
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
EVENT_VOXEL_DATA_ARCHIVE=/workspace/event-voxel-real-data.tar.gz \
  ./scripts/runpod/run_all.sh --smoke-only
```

`run_all.sh` checks the environment, working tree, and dataset hashes, builds
the Release CUDA preset, runs CTest and the `[cuda]` suite, then compares every
real benchmark case against the CPU reference before timing it. It runs the
smoke and full matrices in that order. Use `--smoke-only` for the first pod
validation; run without it for the full matrix after reviewing smoke output.
Any failed step stops the script. CUDA tests skip without a device during local
compilation; `run_all.sh` additionally runs
`event_voxel_run_cuda`, which fails if no CUDA device is usable.
The extracted inputs stay at `/workspace/benchmark-datasets`, so a later full
run can simply use `./scripts/runpod/run_all.sh` on the same pod.

Each matrix creates a new directory under `smoke/` or `full/` with the exact
config, raw `measurements.csv`, `summary.csv`, Git and host metadata, GPU query,
environment output, and correctness output. The directory name includes UTC
time, GPU name, commit prefix, and a random suffix. A failed matrix keeps its
partial CSV and marks its metadata as failed. Do not use a failed run for
performance claims.

## Preserve results before ending the pod

After the script succeeds, it prints the path and SHA-256 of a result archive
under `/workspace`. Transfer that archive to local storage, verify its hash,
and inspect the CSV files. Only then terminate the paid pod. For example:

```bash
scp -i <private-key> -P <ssh-port> \
  root@<pod-host>:<printed-archive-path> ./
sha256sum <downloaded-archive>
tar -tzf <downloaded-archive> | head
```

The project creates the archive but does not transfer it automatically.

## Local verification

The CPU test preset and CUDA compile preset can be validated without a GPU.
Actual CUDA numerical behavior, benchmark latency, and the container image
must be verified on a GPU pod. The first pod session should be treated as a
pipeline validation run before relying on full thesis measurements.
