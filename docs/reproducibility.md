# Reproducing the CPU study

## Software and source

The tested local Python environment is Python 3.13.2 with exact package
versions in `requirements.txt`. `make setup` creates `.venv` and installs them.
The CI pipeline also checks a clean install on Ubuntu. CMake FetchContent pins
Catch2 and Google Benchmark to exact commits. `scripts/setup_external.sh`
checks out exact ECF and RPG revisions; `validate_outputs.py` additionally
checks the RPG source-file SHA-256.

The raw recordings and nested external checkouts are not stored in this Git
repository. The dataset catalog and inspection reports give their names and
SHA-256 fingerprints. A different raw fingerprint defines a different dataset
revision and must get a new inspection report.
GitHub Actions runs CPU Debug, Release, and sanitizer tests plus the synthetic
HDF5-to-NumPy pipeline. The raw-recording verification gate runs locally with
`make verify-coverage`, because the raw recordings are outside Git and CI.

## From source to a verified result

```bash
make setup
make setup-external
make setup-ecf
make all
make test-release
make test-python
make data-inspect-all
make verify-all
```

For one experiment, prepare a named full recording or selected window, verify
the chosen bins, and run `make benchmark-checked`. Each benchmark invocation
gets a unique UTC timestamp and random run ID. The runner refuses a dirty Git
tree by default. Commit the source and small reports before formal thesis runs.

## Artifact contract

- `data/raw/`: untouched original HDF5 files, outside Git.
- `data/processed/`: versioned little-endian `EVXEVT01` binary batches plus
  selection and binary-hash metadata, outside Git.
- `results/metadata/`: source validation reports, retained.
- `results/validation/`: external and independent float64 voxel comparisons,
  retained.
- `results/benchmarks/`: raw measurements, manifest, log, and validation copies,
  retained or archived immutably.
- `results/reports/`: summaries across independent processes, retained.
- `results/voxels/` and `results/visualizations/`: regenerable arrays and figures,
  outside Git unless a figure is selected for publication.

The binary header is 40 bytes: eight-byte magic, little-endian `uint32`
width/height, `uint64` event count, and `int64` start/end timestamps. Each
event is 16 bytes: little-endian `uint16` x/y, `int16` polarity, two zero
reserved bytes, and `int64` timestamp. The end timestamp is exclusive.
