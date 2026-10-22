#pragma once

#include "event_voxel/event_window.hpp"

#include <cstddef>

namespace event_voxel::data {

// A borrowed, decoded CD window. SDK-backed implementations own the slice lifetime.
class TimeSliceSource {
  public:
    virtual ~TimeSliceSource() = default;
    virtual bool next(EventWindowView& window) = 0;
    [[nodiscard]] virtual std::size_t peak_window_events() const noexcept = 0;
};

} // namespace event_voxel::data
