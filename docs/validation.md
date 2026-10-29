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

The experiment command measures one backend; it does not compute a CPU
reference in a CUDA build. The C++ tests compare CUDA strategies with the CPU
reference on generated windows. Run the HDF5 integration test with
`EVENT_VOXEL_HDF5_TEST_INPUT` to check streaming and window continuity on an
actual recording. For larger-scale numerical checks before RunPod measurements,
extend that test suite with representative windows from the recording.
