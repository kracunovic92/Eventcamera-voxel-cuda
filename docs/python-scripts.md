# Python scripts and their place in the pipeline

The two Make entry points are `make pipeline-local` and `make pipeline-runpod`.
Use `SMOKE=1` with either one for a small workflow check. The local pipeline
starts with HDF5 files and produces a verified dataset archive. The RunPod
pipeline starts with that archive and produces a result archive.

```text
Local:  raw HDF5 → prepared events.bin → CPU grids → RPG and float64 checks
        → CPU measurements → dataset archive
RunPod: dataset archive → prepared events.bin → CPU and CUDA grids → RPG and float64 checks
        → CUDA measurements → result archive
```

## `tools/`: local data and CPU measurements

| Script | What it does | Why it exists |
|---|---|---|
| `inspect_dataset.py` | Reads raw HDF5 structure, statistics, and source hash into a metadata report. | Checks the source recording before conversion. |
| `prepare_dataset.py` | Converts a declared recording window into `events.bin` and `metadata.json`. | Makes a small, versioned, hashable input that C++ and RunPod can load without HDF5. |
| `generate_synthetic.py` | Writes controlled synthetic `events.bin` workloads. | Tests behavior such as contention independently of camera recordings. |
| `summarize_validation.py` | Checks that the expected recordings and bin counts have valid inspection, preparation, and verification reports. | Provides a coverage report before a formal CPU matrix. |
| `run_benchmark.py` | Runs one Google Benchmark CPU process and saves its JSON, log, and provenance manifest. | Keeps each CPU timing run reproducible. |
| `run_benchmark_matrix.py` | Schedules independent CPU benchmark processes from a JSON config. | Measures multiple recordings and rounds without manually launching every case. |
| `summarize_benchmarks.py` | Verifies CPU run artifacts and summarizes medians across independent processes. | Produces the CPU performance report without pooling unrelated iterations. |

## `scripts/experiments/`: shared capture and RunPod measurements

| Script | What it does | Why it exists |
|---|---|---|
| `package_datasets.py` | Checks config hashes and archives selected prepared inputs. | Transfers only the exact data a pod needs. |
| `capture_real.py` | Runs `event_voxel_run_cpu` for each configured case and also `event_voxel_run_cuda` when supplied. | Saves full voxel grids and one-run JSON records for later checks. |
| `validate_outputs.py` | Checks one saved grid or a captured run against RPG, the independent float64 oracle, or both; its `compare-backends` mode writes a CPU versus GPU CSV. | One correctness command for local CPU and RunPod CPU/CUDA experiments, with an optional backend diagnostic. |
| `reference_models.py` | Contains the pinned RPG adapter and the float64 triangular-weight implementation. | Keeps the independent reference calculations out of the command and available to both modes. |
| `output_comparison.py` | Reads and checks saved `.npy` and JSON artifacts, and implements CPU versus GPU array comparison. | Shared code for optional cross-backend diagnostics and artifact integrity checks. |
| `run_matrix.py` | Calls the C++ CUDA benchmark for every configured dataset, count, bin count, and strategy; retains each timing row and machine metadata. | Orchestrates repeated GPU measurements without putting the experiment schedule into C++. |
| `summarize_results.py` | Calculates descriptive statistics from the CUDA timing CSV. | Preserves raw measurements while providing a compact report. |

The shell scripts `pipeline_local.sh` and `pipeline_runpod.sh` put these steps in
order. `scripts/runpod/prepare_datasets.py` downloads or unpacks the dataset
archive and verifies its hashes; it can extend a smoke dataset directory with
the later full archive. `scripts/runpod/verify_environment.sh` checks the pod
toolchain and GPU before a run.

`tests/python/` contains automated checks for the HDF5 pipeline, synthetic
data, the float64 oracle, saved-output comparison, and external-reference
sampling. Test scripts do not create experiment results.

The CPU benchmark config currently covers first-100k recording windows at
2, 5, and 10 bins. The CUDA full config also covers other event counts,
20 bins, and a larger sparklers selection. Direct speedups should only compare
matching dataset, selected event count, bin count, and timing boundary; the
remaining CUDA cases do not yet have repeated CPU benchmark counterparts.
