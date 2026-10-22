# Numerical validation

For an event `(x, y, t, p)` in `[t0, t1)`, polarity `p=1` contributes `+1` and
`p=0` contributes `-1`. Its temporal position is
`u=(B-1)(t-t0)/(t1-t0)`. Bin `b` receives the signed triangular weight
`max(0, 1-|b-u|)`. The output is contiguous `float32` in `[B,H,W]` order.

The CPU reference computes the temporal position in `double`, converts each
weight to `float`, and accumulates in `float`. CUDA may reorder contended
additions, so compare with a numerical tolerance rather than bitwise equality.

Run the C++ tests for boundaries, interpolation, polarity, empty and dense
windows, and CUDA strategy agreement when a GPU is present:

```bash
cmake --preset cuda-release
cmake --build --preset cuda-release --parallel
ctest --preset cuda-release --output-on-failure
```

For an experiment-level check, use `--mode validate` with the same input,
window duration, and bin count planned for measurement. It checks one selected
window: `--export-index N` chooses that window, starting at zero. When a GPU is
available it compares all three CUDA strategies with the CPU output; otherwise
it checks the CPU path. Repeat selected indices to cover different event rates.
The command exits with an error when a comparison fails. Save its successful
command output alongside the result folder before measuring.

```bash
./build/cuda-release/event_voxel_experiment \
  --input data/raw/sparklers.hdf5 --mode validate \
  --window-us 50000 --results-root results
```

With Metavision SDK Stream installed, also build and run the HDF5 integration
test using `EVENT_VOXEL_HDF5_TEST_INPUT`. It checks bounded streaming and
window continuity on an actual recording.
