# Dataset catalog

All seven files in `data/raw/` passed the chunked HDF5 inspection at the
current source fingerprints. Inspection checks geometry, field types,
coordinates, polarities, nonnegative timestamps, and timestamp order.
The first 100,000 events from each source were also checked against RPG and an
independent float64 oracle for 2, 5, and 10 bins. This sample validation does
not cover every event in the full recordings. Run `make data-inspect-all` to
regenerate source fingerprints in `results/metadata/`.

| Recording | Resolution | Events | Timestamp range (µs) |
|---|---:|---:|---:|
| 80_balls | 640×480 | 4,588,809 | 54–6,284,062 |
| driving_sample | 1280×720 | 111,484,516 | 4,096–12,500,991 |
| eye_tracking | 1280×720 | 56,844,143 | 0–9,893,387 |
| hand_spinner | 640×480 | 11,965,597 | 16–4,999,999 |
| sparklers | 640×480 | 521,252 | 0–95,871 |
| spinner | 640×480 | 54,165,303 | 0–5,002,329 |
| traffic_monitoring | 640×480 | 17,939,153 | 48–27,999,870 |

The full `driving_sample` recording alone requires about 2.68 GB for the
current 24-byte in-memory C++ `Event` objects, before output and other copies.
Use bounded selections for routine cross-dataset verification. A full-recording
run is a distinct experiment with its own memory requirements.

## Selecting a reproducible window

```bash
make data-prepare DATASET=driving_sample WINDOW_ID=first_100k WINDOW_MAX_EVENTS=100000
make data-prepare DATASET=driving_sample WINDOW_ID=us_500000_550000 \
  WINDOW_START_US=500000 WINDOW_END_US=550000
```

The `WINDOW_ID` is a human label; `metadata.json` records the actual selection
and SHA-256 of both source and binary. Keep distinct labels for distinct
selections. `make verify-all` prepares the same first 100,000 events from each
recording and checks all configured bin counts. It does not require preparing
all 111 million driving events.

`WINDOW_START_INDEX` selects by source event position, which is useful for
middle and tail samples without choosing timestamps by hand. For example,
`WINDOW_START_INDEX=500000 WINDOW_MAX_EVENTS=100000` selects 100,000 events
starting at source event 500,000. When no temporal start is given, the prepared
time origin is the first selected timestamp. `make verify-coverage` computes
first, middle, and tail indices for every recording and verifies each sample.
