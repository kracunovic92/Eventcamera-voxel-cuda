#include "experiment.hpp"

#include <iomanip>
#include <string>

namespace event_voxel::experiment {

void Experiment::record_input_window(const EventWindowView& window, double preparation_ms) {
    ++stats_.windows;
    stats_.total_events += window.events.size();
    stats_.window_event_counts.push_back(static_cast<double>(window.events.size()));
    stats_.input_preparation_times_ms.push_back(preparation_ms);
    if (preparation_ms > double(config_.window_us) / 1000.0)
        ++stats_.input_preparation_over_interval_windows;
    if (stats_.windows == 1)
        stats_.first_window_start_us = window.start_us;
    stats_.last_window_end_us = window.end_us;
    stats_.sensor_width = window.width;
    stats_.sensor_height = window.height;
    files_.windows << window.sequence << ',' << window.start_us << ',' << window.end_us << ','
                   << window.events.size() << ',' << std::setprecision(12)
                   << (1'000'000.0 * window.events.size() / (window.end_us - window.start_us))
                   << ',' << window.partial_final << ',' << preparation_ms << '\n';
}

void Experiment::record_sample(const EventWindowView& window, std::string_view backend,
                               std::string_view stage, double wall_ms,
                               std::optional<double> cuda_event_ms, std::size_t bytes) {
    files_.samples << window.sequence << ',' << backend << ',' << stage << ','
                   << std::setprecision(12) << wall_ms << ',';
    if (cuda_event_ms)
        files_.samples << *cuda_event_ms;
    files_.samples << ',' << bytes << '\n';
    stats_.stage_wall_times_ms[std::string(backend) + "/" + std::string(stage)].push_back(wall_ms);
}

} // namespace event_voxel::experiment
