# Event Voxel CUDA

C++20 experiments for converting event-camera streams into temporal voxel grids.
The experiment reads fixed-duration windows directly from a Metavision HDF5
recording. It measures a CPU reference and three CUDA strategies without an
intermediate data format.

## Build and test

All builds need CMake 3.30+, Ninja, a C++20 compiler, Git, and Metavision SDK
Stream with HDF5 support. CMake fetches pinned Catch2 source when tests are
enabled. A recording is only needed to run the experiment or its optional
HDF5 integration test. If CMake cannot find the SDK, set
`MetavisionSDK_DIR` to the directory containing its CMake package.
This checkout also discovers a project-local OpenEB install at `.deps/openeb`.
In VS Code, select and configure a CMake preset after installing the SDK so
IntelliSense reads the generated compile commands with SDK include paths.

```bash
cmake --preset local-release
cmake --build --preset local-release --parallel
ctest --preset local-release --output-on-failure
```

On a machine with an NVIDIA GPU and CUDA development toolkit, the CUDA build
also runs the unit tests:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
```

`make test` is a shortcut for the selected CMake preset. Unit tests use small
hand-written event fixtures; the experiment CLI accepts only HDF5 input.
For a local CUDA compile check without a GPU, select `cuda-local-check`; it
uses the `~/miniconda3/envs/cuda-dev` toolkit. The `cuda-release` preset is for
a machine with `nvcc` on `PATH` and an NVIDIA GPU.

## Real recordings

All presets link Metavision SDK Stream. The integration test can read a
local recording when `EVENT_VOXEL_HDF5_TEST_INPUT` points to it.

```bash
cmake --preset cuda-release \
  -DMetavisionSDK_DIR=<SDK-cmake-directory> \
  -DEVENT_VOXEL_HDF5_TEST_INPUT=<recording.hdf5>
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
./build/cuda-release/event_voxel_experiment --input <recording.hdf5>
```

This runs the default `pipeline` measurement with 100 ms windows and five
temporal bins. On a GPU host, `--scenario auto` measures all four processing
paths; on a CPU-only host, it measures `cpu-host`. Use `--window-us`, `--bins`,
and `--strategy` to vary the main experiment parameters. `--help` lists the
less common controls.

`--mode` chooses the measurement: `pipeline` for complete ready-window work,
`kernel` for the final CUDA kernel, `strategy` for clear/preprocessing/kernel,
`transfer` for memory copies, `replay` for a simulated queue, `profile` for one
window, `validate` for correctness, or `export` for `.npy` output. `--scenario`
chooses where processing runs and where its result ends up: `cpu-host`,
`cpu-to-device`, `gpu-device`, or `gpu-to-host`. The `auto` default selects the
available paths. GPU-only modes select `gpu-device` automatically.

`--window-us` sets the duration of each SDK time slice and is a research
parameter. `--max-window-events` rejects slices that exceed the configured
event count.

Each invocation creates a `results/run-*/` directory with per-window timing,
sample measurements, metadata, and a summary. `--mode validate` compares CUDA
strategies with the CPU reference; `--mode export` can save selected `.npy`
grids. See [streaming experiments](docs/streaming-experiments.md) for all modes
and timing boundaries.

## Guides

- [Architecture](docs/architecture.md): input, windowing, CPU, and CUDA connections.
- [Datasets](docs/datasets.md): raw recording catalog and window choices.
- [Validation](docs/validation.md): numerical rule and correctness checks.
- [Reproducibility](docs/reproducibility.md): source and result provenance.
- [RunPod](docs/runpod.md): GPU setup and network-volume transfers.
- [Research design](docs/research-design.md): questions and experiment dimensions.
