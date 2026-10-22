#include "experiment.hpp"

#include "event_voxel/data/metavision_hdf5.hpp"

namespace event_voxel::experiment {
std::unique_ptr<data::TimeSliceSource> open_source(const Config& config) {
    return data::open_metavision_hdf5(config.input, config.window_us, config.max_window_events);
}

} // namespace event_voxel::experiment
