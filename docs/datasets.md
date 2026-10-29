# Dataset catalog

Original recordings belong in ignored `data/raw/` or another local or mounted
directory. The experiment reads them directly through Metavision SDK Stream.
The following counts are from earlier source inspections; check the recording
and its SHA-256 hash before a formal run.

| Recording | Resolution | CD events | Timestamp range (µs) |
|---|---:|---:|---:|
| 80_balls | 640×480 | 4,588,809 | 54–6,284,062 |
| driving_sample | 1280×720 | 111,484,516 | 4,096–12,500,991 |
| eye_tracking | 1280×720 | 56,844,143 | 0–9,893,387 |
| hand_spinner | 640×480 | 11,965,597 | 16–4,999,999 |
| sparklers | 640×480 | 521,252 | 0–95,871 |
| spinner | 640×480 | 54,165,303 | 0–5,002,329 |
| traffic_monitoring | 640×480 | 17,939,153 | 48–27,999,870 |

Choose `--window-us` to define each SDK time slice. The experiment processes
all slices through the end of the file. The SDK queue holds at most one
completed slice, but the number of events in a slice is not capped. Record the
source file hash with each result archive; a path alone does not identify a
dataset revision.
