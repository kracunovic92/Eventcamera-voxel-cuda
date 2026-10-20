#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "event_voxel/data/window_config.hpp"
#include "event_voxel/event_window.hpp"

namespace event_voxel::data {

// Reader implementations expose decoded CD events in original timestamp order.
// next() must not buffer the whole recording.
class EventReader {
  public:
    virtual ~EventReader() = default;
    virtual bool next(Event& event) = 0;
    [[nodiscard]] virtual std::uint32_t width() const = 0;
    [[nodiscard]] virtual std::uint32_t height() const = 0;
};

class EventWindowSource {
  public:
    EventWindowSource(std::unique_ptr<EventReader> reader, WindowConfig config = {});
    // The returned span is invalidated by the next call, including a call returning false.
    bool next(EventWindowView& window);
    [[nodiscard]] std::size_t peak_buffered_events() const noexcept {
        return peak_buffered_events_;
    }

  private:
    bool refill();
    bool peek(Event& event);
    std::unique_ptr<EventReader> reader_;
    WindowConfig config_;
    std::vector<Event> chunk_;
    std::vector<Event> staging_;
    std::optional<Event> pending_;
    std::size_t position_ = 0;
    std::size_t peak_buffered_events_ = 0;
    std::int64_t start_us_ = 0;
    std::int64_t last_us_ = 0;
    std::uint64_t sequence_ = 0;
    bool started_ = false;
    bool eof_ = false;
};

} // namespace event_voxel::data
