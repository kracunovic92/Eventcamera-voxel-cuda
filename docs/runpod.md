# RunPod event-window experiments

Run `event_voxel_experiment` on a GPU pod. The experiment reads original HDF5
recordings through Metavision SDK Stream and its ECF codec. It needs a CUDA
development toolkit and NVTX3 headers for CUDA builds.

## Direct SSH workflow for the current pod

Use a local terminal in this repository. Copy the current **SSH over exposed
TCP** host and port from the pod's Connect tab; the port can change when the
pod is restarted. Use the private key whose public key is registered with
RunPod. The scripts assume the checkout is at
`/workspace/Eventcamera-voxel-cuda` and that both `local-release` and
`cuda-release` experiment binaries are already built there.

```bash
export RUNPOD_SSH_TARGET=root@HOST_FROM_CONNECT
export RUNPOD_SSH_PORT=PORT_FROM_CONNECT
export RUNPOD_SSH_KEY="$HOME/.ssh/id_rsa" # omit if this is already your key

./scripts/runpod/upload_raw.sh
./scripts/runpod/run_benchmarks.sh
./scripts/runpod/download_results.sh
```

`upload_raw.sh` copies everything in local `data/raw/` into the pod checkout's
`data/raw/`. By default, `run_benchmarks.sh` processes each `.hdf5` recording
with CPU, CUDA event-scatter, pixel-gather, and sort-reduce for every
combination of 50, 100, 200, and 500 ms windows and 2, 5, and 10 bins.
CUDA block size is 256. Seven recordings produce 336 runs. To select a
smaller matrix or repeat it, use comma-separated values:

```bash
./scripts/runpod/run_benchmarks.sh --window-ms 50,100 --bins 2,5 \
  --block-size 256 --repetitions 3
```

The original positional form `[window_us] [bins] [block_size] [repetitions]`
still runs a single duration and bin count. Each invocation creates a separate
`/workspace/experiment-results/batch-*` folder with run logs, `runs.tsv`,
`batch-info.txt`, and the experiment's CSV and JSON output. It stops on the
first failed run and keeps partial output for inspection.

`download_results.sh` copies the entire `/workspace/experiment-results/`
tree into local ignored `results/runpod/`, including older runs. It never
deletes results on either machine. Running it again copies files with the
same names again. Pass a batch name to download only one batch, for example
`./scripts/runpod/download_results.sh batch-20261029T152527Z-sExTBy`.
If you use a separate known-hosts file, set
`RUNPOD_KNOWN_HOSTS` to its path before invoking the scripts.

## Inspect timing and publish a batch

Each `results/runpod/batch-*/run-*/` folder has `samples.csv` with one row per
window and stage, `windows.csv` with the event count and input preparation
time, `summary.json` with stage means, medians, and p95s, and `metadata.json`
with the timing definitions. Match CSV rows by `sequence`. On CUDA runs,
`strategy` includes output clearing, preprocessing, and the final kernels;
`h2d-events` and `d2h-grid` measure memory copies. `pipeline` includes those
three stages, and `processing-total` also includes buffer preparation.
`run_wall_ms` additionally includes GPU setup, input reads, and CSV writing.
`comparison.csv` reports measured input preparation, processing, and the
remaining unisolated run-wall time as `other_run_wall_ms`. These are
nested intervals, so adding all stage times double-counts work. The
`cuda_event_ms` column in `samples.csv` is the device event measurement where
available; `wall_ms` is the host-observed duration. See
[streaming experiments](streaming-experiments.md) for the exact boundaries.

Make cross-run tables, then prepare a compact Git-tracked archive of the
detailed CSV and JSON files:

```bash
./scripts/runpod/summarize_batch.py results/runpod/BATCH_NAME
./scripts/runpod/prepare_publication.sh BATCH_NAME
git add results/published/runpod/BATCH_NAME
git commit -m "Publish RunPod benchmark BATCH_NAME"
git push origin HEAD
```

`comparison.csv` has one row per run; `stage_summary.csv` has one row per
stage, including transferred byte counts and mean CUDA event time. The
publication includes those tables, the run manifest, and a compressed archive
with all per-window CSV/JSON results. Check `SHA256SUMS` after cloning.
`prepare_publication.sh` leaves the original downloaded batch in the ignored
`results/runpod/` directory.

## Prepare storage and a pod

For transfers without SCP, create a [RunPod network volume](https://docs.runpod.io/storage/network-volumes)
in a data center that supports its [S3-compatible API](https://docs.runpod.io/storage/s3-api).
Attach it while deploying the pod; it mounts at `/workspace`. Keep the S3 API
key on the local machine. Upload a raw recording with the AWS CLI:

```bash
aws s3 cp --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  data/raw/sparklers.hdf5 \
  s3://NETWORK_VOLUME_ID/data/raw/sparklers.hdf5
```

The file then appears on the pod at `/workspace/data/raw/sparklers.hdf5`.
Record its local `sha256sum` and verify the same value on the pod. Keep raw
recordings and result archives outside Git.

Choose an existing Ubuntu 24.04-based CUDA development template with `nvcc`
and terminal access. The project also needs CMake 3.30+, Ninja, a C++20
compiler, Git, and Make. Install
missing user-space tools on the pod if needed:

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential ca-certificates git ninja-build nlohmann-json3-dev python3-venv
python3 -m venv /opt/event-voxel-tools
/opt/event-voxel-tools/bin/python -m pip install 'cmake>=3.30,<5'
export PATH="/opt/event-voxel-tools/bin:$PATH"
cmake --version
```

Keep using this terminal for the commands below, or repeat the `export PATH`
line in a new terminal.

Record the exact template image, GPU, driver, and toolkit versions. The pod
needs storage for the source, build tree, input recordings, and results.

## Clone and run unit tests

Commit and push the intended source revision first. In the pod terminal:

```bash
cd /workspace
git clone REPOSITORY_URL Eventcamera-voxel-cuda
cd Eventcamera-voxel-cuda
git checkout COMMIT_SHA
./scripts/runpod/verify_environment.sh
cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
```

Unit tests use generated events internally. Run the HDF5 experiment after
installing the SDK and configuring the HDF5 preset below.

## Run an original HDF5 recording

Install [Metavision SDK Stream and its ECF codec](https://docs.prophesee.ai/stable/installation/linux_openeb_with_packages.html)
on the pod. Supply the generated `MetavisionSDK_DIR` CMake package location and
the raw recording path when configuring. The real-recording adapter requires a
build and integration run on an SDK host before results can be relied on.

```bash
cd /workspace/Eventcamera-voxel-cuda
cmake --preset cuda-release \
  -DMetavisionSDK_DIR=SDK_CMAKE_DIRECTORY \
  -DEVENT_VOXEL_HDF5_TEST_INPUT=/workspace/Eventcamera-voxel-cuda/data/raw/sparklers.hdf5
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
./build/cuda-release/event_voxel_experiment \
  --input /workspace/Eventcamera-voxel-cuda/data/raw/sparklers.hdf5 \
  --window-us 100000 --bins 5 --strategy event-scatter \
  --results-root /workspace/experiment-results
```

Run the CPU build on the same pod for CPU measurements:

```bash
cmake --preset local-release
cmake --build --preset local-release --parallel
./build/local-release/event_voxel_experiment \
  --input /workspace/Eventcamera-voxel-cuda/data/raw/sparklers.hdf5 \
  --window-us 100000 --bins 5 \
  --results-root /workspace/experiment-results
```

Repeat the CUDA command with `pixel-gather` and `sort-reduce`, then with
other durations and recordings after the tests pass. Each invocation processes
the full file once. See
[streaming experiments](streaming-experiments.md) for timing boundaries.

## Retrieve results without a direct pod connection

The network volume exposes `/workspace/experiment-results` through the S3 API.
You can copy a run folder through that API, or archive it first on the pod. For
one run, replace `RUN_NAME` with the printed `run-*` directory name:

```bash
cd /workspace/experiment-results
tar -czf /workspace/RUN_NAME.tar.gz RUN_NAME
sha256sum /workspace/RUN_NAME.tar.gz
```

On the local machine, download the archive and compare `sha256sum` with the
value printed by the pod:

```bash
mkdir -p results/downloads
aws s3 cp --region DATACENTER \
  --endpoint-url https://s3api-DATACENTER.runpod.io/ \
  s3://NETWORK_VOLUME_ID/RUN_NAME.tar.gz results/downloads/
sha256sum results/downloads/RUN_NAME.tar.gz
```

Keep an independent local or personal-drive copy of the verified result and
source hashes. A network volume persists across pod replacements, but it is
still a separate paid storage resource.
