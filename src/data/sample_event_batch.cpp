#include "event_voxel/data/sample_event_batch.hpp"

#include <stdexcept>

namespace event_voxel {

EventBatch sample_evenly(EventBatch source, std::size_t event_count) {
    if (event_count == 0 || event_count > source.events.size()) {
        throw std::invalid_argument("requested event count is outside the recording");
    }
    if (event_count == source.events.size()) {
        return source;
    }

    EventBatch selected{};
    selected.width = source.width;
    selected.height = source.height;
    selected.start_us = source.start_us;
    selected.end_us = source.end_us;
    selected.events.reserve(event_count);

    if (event_count == 1) {
        selected.events.push_back(source.events.front());
        return selected;
    }
    for (std::size_t i = 0; i < event_count; ++i) {
        const auto index = i * (source.events.size() - 1) / (event_count - 1);
        selected.events.push_back(source.events[index]);
    }
    return selected;
}

} // namespace event_voxel
