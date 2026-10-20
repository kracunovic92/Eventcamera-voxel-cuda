# CUDA experiment scripts

The C++ programs in `benchmarks/` perform the timed work. These scripts select
cases, save results, and analyze files. They do not implement voxelization.

| Pipeline | Command | Stages |
|---|---|---|
| Local CPU | `make pipeline-local` | Prepare HDF5 inputs; build and test; validate CPU outputs with RPG and the float64 oracle; capture the configured CPU grids; validate them with both references; run the independent CPU benchmark matrix; summarize; package the inputs for RunPod. |
| RunPod CUDA | `make pipeline-runpod` | Verify the transferred inputs; build and test; capture CPU and CUDA grids; validate each grid with both references; run the CUDA benchmark matrix; summarize and archive results. |

Add `SMOKE=1` to either Make command for its smaller smoke configuration. The
local pipeline requires `.venv` and the ECF HDF5 decoder. The pod needs the
dataset archive produced locally. Both pipelines use the pinned RPG revision.
Their timing protocols differ: CPU uses independent Google Benchmark processes;
CUDA records repeated synchronous end-to-end calls.

| Script | Role | Main output |
|---|---|---|
| `package_datasets.py` | Verify and archive prepared inputs for transfer to RunPod. | Local session `datasets.tar.gz` |
| `capture_real.py` | Run the CPU app and, on RunPod, the CUDA app once per configured case. | `.npy` grids, JSON records, `manifest.json` |
| `validate_outputs.py` | Validate one saved grid or every grid in a captured run against RPG and/or the float64 oracle; optionally compare saved CPU and GPU grids with `compare-backends`. | Combined JSON report, `validation.csv`, or `comparisons.csv` |
| `reference_models.py` | Implement the pinned RPG adapter and independent float64 oracle used by the validator. | None directly |
| `output_comparison.py` | Shared array and run-record reader for the comparison scripts. | None directly |
| `run_matrix.py` | Verify inputs and execute the C++ CUDA benchmark across the configured matrix. | `measurements.csv`, run metadata |
| `summarize_results.py` | Group saved benchmark measurements and calculate descriptive statistics. | `summary.csv` |
| `pipeline_local.sh` | Execute the local CPU experiment sequence. | A local session under `results/experiments/local/` |
| `pipeline_runpod.sh` | Execute the RunPod CUDA experiment sequence. | A pod session and downloadable archive |

The optional `validate_outputs.py compare-backends` compares
our two backends; `validate_outputs.py capture` is the correctness gate. See [the RunPod runbook](../../docs/runpod.md)
for the command sequence and data transfer instructions.

The apps' `voxelize_ms` is one synchronous host-clock call. The C++ CUDA
benchmark also uses a host clock but repeats the call after warmups and records
each `end_to_end_ms` sample. Neither value is kernel-only device time.
