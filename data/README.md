# Data

Place the seven original Prophesee HDF5 files in `data/raw/`. Raw recordings
and prepared binaries are ignored by Git. Their names, geometry, event counts,
and source hashes are listed in [the dataset catalog](../docs/datasets.md).

`data/processed/<dataset>/events.bin` holds a full prepared recording.
`data/processed/<dataset>/<window-id>/events.bin` holds a selected window.
Each folder also contains `metadata.json` with the source hash, selection,
temporal bounds, event count, and prepared-binary hash. Never identify an
experiment only by its folder name; check the metadata and binary hash.
