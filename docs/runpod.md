# RunPod event-window experiments

Run `event_voxel_experiment` on a GPU pod. The experiment reads original HDF5
recordings through Metavision SDK Stream and its ECF codec. It needs a CUDA
development toolkit for GPU scenarios.

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
  build-essential ca-certificates git ninja-build python3-venv
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
git clone REPOSITORY_URL event-voxel-cuda
cd event-voxel-cuda
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
cd /workspace/event-voxel-cuda
cmake --preset cuda-release \
  -DMetavisionSDK_DIR=SDK_CMAKE_DIRECTORY \
  -DEVENT_VOXEL_HDF5_TEST_INPUT=/workspace/data/raw/sparklers.hdf5
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
./build/cuda-release/event_voxel_experiment \
  --input /workspace/data/raw/sparklers.hdf5 --mode validate \
  --window-us 100000 --bins 5 --export-index 0 \
  --results-root /workspace/experiment-results
./build/cuda-release/event_voxel_experiment \
  --input /workspace/data/raw/sparklers.hdf5 --mode pipeline --scenario all \
  --window-us 100000 --bins 5 --max-windows 100 --warmup 3 --repeat 10 \
  --results-root /workspace/experiment-results
```

Validate more window indices with `--export-index N` to cover varied event
rates. Repeat the pipeline command with other durations and recordings only
after the integration and validation runs pass. See [streaming experiments](streaming-experiments.md)
for timing boundaries and memory caps.

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
