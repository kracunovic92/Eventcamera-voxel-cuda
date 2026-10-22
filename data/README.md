# Data

Place original Metavision `.hdf5` recordings in `data/raw/`, or provide an
absolute path through `--input`. Raw files are ignored by Git and read directly
by `event_voxel_experiment` when Metavision SDK Stream and its ECF codec are
installed. The experiment does not require a converted input format.

Record a SHA-256 for each file used in a formal run. See the
[dataset catalog](../docs/datasets.md) for known recordings and window choices.
