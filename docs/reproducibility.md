# Reproducing an event-window experiment

Use a committed source revision and record the exact SHA-256 of each input HDF5
file. Raw recordings and generated result folders are outside Git.

```bash
git rev-parse HEAD
sha256sum data/raw/sparklers.hdf5
cmake --preset cuda-release \
  -DMetavisionSDK_DIR=<SDK-cmake-directory> \
  -DEVENT_VOXEL_HDF5_TEST_INPUT=data/raw/sparklers.hdf5
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
./build/cuda-release/event_voxel_experiment \
  --input data/raw/sparklers.hdf5 --mode pipeline --scenario all \
  --window-us 100000 --bins 5 --max-windows 100 --warmup 3 --repeat 10
```

Each `results/run-*/` folder contains `metadata.json` with configuration,
build and Git identity, source path, and available hardware details. Save the
exact command too: some CLI limits, including `--max-windows`, are not recorded
in metadata. `windows.csv`
records the windows and input preparation time. `samples.csv` contains every
timed sample. `summary.json` aggregates samples, and `crossovers.csv` compares
CPU and GPU scenarios where both exist. Save the input hash and exact SDK,
CUDA toolkit, compiler, template or image versions alongside that folder.

For performance comparisons, use the same recording, time windows, bin count,
GPU, build type, and scenario. Run correctness mode before reporting a new CUDA
strategy, and archive the raw result folder before ending a temporary pod. See
[streaming experiments](streaming-experiments.md) for what each timing includes.
