# RunPod CUDA benchmark runbook

## Scope

The first GPU session validates the CUDA build, strategy correctness against
the pinned RPG implementation, smoke matrix, and raw result capture. The
current benchmark reports host end-to-end latency for each
`CudaVoxelizer::voxelize()` call. It includes allocation,
host-to-device transfer, computation, device-to-host transfer, and cleanup.
The call synchronizes before returning, so a host clock includes completed GPU
work. It does not report kernel-only GPU time. CUDA phase timing and
GPU-resident or buffer-reuse modes are future work.

The committed smoke and full matrices use prepared real-camera recordings.
Each input path and SHA-256 hash is fixed in the configs. The benchmark loads
the binary outside the timed region and selects evenly spaced, ordered events
across the recorded window for each event count. All strategies receive the
same selected events. Raw HDF5 recordings stay on the local machine.

## Transfer the real-data inputs

The prepared binaries are ignored by Git. The local experiment pipeline
prepares them, validates the CPU outputs with the external RPG implementation,
and produces a transferable archive:

```bash
make setup
make setup-ecf
make pipeline-local SMOKE=1
```

The smoke run prints its session directory and a `datasets.tar.gz` path. For
the full input archive, run `make pipeline-local` after committing the source.
The packager verifies every binary against its committed hash and
`metadata.json`, then prints the archive hash. Keep the original HDF5 files
and the archive in local storage. Transfer the archive to the pod using the
pod's SSH address and port, for example:

```bash
scp -i <private-key> -P <ssh-port> <local-session>/datasets.tar.gz \
  root@<pod-host>:/workspace/event-voxel-real-data.tar.gz
```

Alternatively, put the archive at an HTTPS URL accessible from the pod. Set
`EVENT_VOXEL_DATA_URL` to that URL. The run script downloads the archive,
extracts it outside Git, and verifies every binary hash before building or
benchmarking. `SMOKE=1` accepts the smaller smoke archive; a full run needs the
full archive. The preparation script can add the remaining verified datasets
to the same pod directory when you later supply that full archive. A persistent
network volume can hold the extracted dataset root for later pods;
`EVENT_VOXEL_DATA_ROOT` changes its location.

```bash
EVENT_VOXEL_DATA_URL='<private-https-url>' make pipeline-runpod SMOKE=1
```

## Prepare a known revision

Build and publish `infra/runpod/Dockerfile` to a registry accessible to your
RunPod account, using the repository root as Docker build context. The image
uses `nvidia/cuda:12.8.1-devel-ubuntu24.04` and installs CMake, Ninja, a C++
compiler, Python 3 with NumPy, Git, certificates, and OpenSSH. Add your public
SSH key to your RunPod account and expose TCP port 22 when creating the pod. The startup
script receives `PUBLIC_KEY`, starts SSH, and enables SCP transfer. The pipeline
checks out the pinned RPG source on the pod.
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
  make pipeline-runpod SMOKE=1
```

`pipeline_runpod.sh` checks the environment, working tree, and dataset hashes,
builds the Release CUDA preset, and runs CTest, including CUDA cases. Before
each benchmark matrix, `capture_real.py` runs `event_voxel_run_cpu` and
`event_voxel_run_cuda` for every dataset, event count, bin count, and CUDA
strategy. Each app saves a complete float32 `.npy` grid and a JSON run record.
`validate_outputs.py capture` reads those arrays and computes the pinned RPG
and independent float64 references from the same selected events. It writes
`validation.csv` and stops the run if any CPU or GPU grid disagrees. The optional
`validate_outputs.py compare-backends` command compares saved GPU arrays to saved CPU arrays
without rerunning voxelization. The runner then measures the smoke and full
matrices in that order. Use `SMOKE=1` for the first pod validation; run without
it for the full matrix after reviewing smoke output. Any failed step
stops the script. CUDA tests skip without a device during local compilation;
the CUDA app fails if no device is usable.
The extracted inputs stay at `/workspace/benchmark-datasets`; supply the full
archive before a later `make pipeline-runpod` on the same pod.

Each invocation creates a session directory under `EVENT_VOXEL_RESULT_ROOT`.
It contains `data/outputs/smoke/` and `data/outputs/full/` with captured grids,
run records, manifests, and validation CSVs. The `smoke/` and `full/`
benchmark directories contain the exact config, raw `measurements.csv`, `summary.csv`,
Git and host metadata, GPU query, environment output, and correctness output.
The benchmark directory name includes UTC time, GPU name, commit prefix, and a
random suffix. A failed matrix keeps its partial CSV and marks its metadata as
failed. Do not use a failed run for performance claims.

The full matrix saves about 5.76 GiB of uncompressed voxel arrays before the
result archive is compressed. Allow room for the arrays, archive, and benchmark
results on the pod. Single app run records include input loading, one
voxelization call, NumPy export time, process peak RSS, and GPU memory free
before and after the call. GPU free memory snapshots are not peak allocation
measurements. The separate benchmark CSV is the source for repeated latency
statistics; NumPy export and file loading are outside its timed region.
In app run records, `voxelize_ms` is one host-clock timing of the synchronous
voxelization call. It is diagnostic, not a kernel-only or repeated benchmark
measurement.

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
After extracting it locally, inspect the archived `validation.csv`
for direct RPG and float64 checks. You can also run
`python3 scripts/experiments/validate_outputs.py compare-backends <session>/data/outputs/full/<capture-id>`
to compare saved GPU and CPU arrays as a secondary diagnostic. That command
uses only the saved arrays and Python standard library; the absolute pod paths
in per-app run records are provenance and are not needed after transfer. To
regenerate the validation, provide the extracted `events.bin` root, the
pinned RPG checkout, and NumPy to `validate_outputs.py capture`.

## Local verification

The CPU test preset and CUDA compile preset can be validated without a GPU.
Actual CUDA numerical behavior, benchmark latency, and the container image
must be verified on a GPU pod. The first pod session should be treated as a
pipeline validation run before relying on full thesis measurements.
