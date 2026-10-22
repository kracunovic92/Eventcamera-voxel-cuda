#include "event_voxel/data/metavision_hdf5.hpp"

#include <metavision/sdk/stream/camera_stream_slicer.h>
#include <metavision/sdk/stream/file_config_hints.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef EVENT_VOXEL_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace event_voxel::data {
namespace {

class MetavisionHdf5TimeSliceSource final : public TimeSliceSource {
  public:
    MetavisionHdf5TimeSliceSource(const std::filesystem::path& path, std::int64_t window_us,
                                  std::size_t max_window_events)
        : slicer_(open_camera(path),
                  Metavision::CameraStreamSlicer::SliceCondition::make_n_us(window_us), 1),
          window_us_(window_us), max_window_events_(max_window_events),
          width_(slicer_.camera().geometry().get_width()),
          height_(slicer_.camera().geometry().get_height()) {
        if (window_us_ <= 0 || max_window_events_ == 0 || width_ == 0 || height_ == 0 ||
            width_ > 65'536 || height_ > 65'536)
            throw std::invalid_argument("Invalid time slice duration, event cap or geometry");
    }

    bool next(EventWindowView& window) override {
        if (!started_) {
            iterator_ = slicer_.begin();
            started_ = true;
        } else if (iterator_ != slicer_.end()) {
            ++iterator_;
        }
        // A recording can have trigger-only time before its first CD event.
        while (iterator_ != slicer_.end() && sequence_ == 0 &&
               (!iterator_->events || iterator_->events->empty()))
            ++iterator_;
        if (iterator_ == slicer_.end()) {
            converted_.clear();
            return false;
        }

        const auto& slice = *iterator_;
        if (slice.status != Metavision::Slice::ConditionStatus::MET_N_US && slice.events &&
            slice.events->empty()) {
            converted_.clear();
            return false;
        }
        if (!slice.events || slice.n_events != slice.events->size() ||
            slice.n_events > max_window_events_)
            throw std::runtime_error("SDK slice is invalid or exceeds max_window_events");

        // For MET_N_US, t is the exclusive upper bound. For the final flush,
        // t is the last event timestamp, so extend it by one microsecond.
        const bool full = slice.status == Metavision::Slice::ConditionStatus::MET_N_US;
        const std::int64_t end_us = full ? slice.t : checked_last_end(slice.t, slice.n_events);
        const std::int64_t start_us =
            sequence_ == 0 ? first_start(slice, end_us, full) : next_start_us_;
        if (end_us <= start_us || end_us - start_us > window_us_ ||
            (full && end_us - start_us != window_us_))
            throw std::runtime_error("Unexpected SDK time slice boundary");

#ifdef EVENT_VOXEL_NVTX
        nvtx3::scoped_range marker("convert CD slice");
#endif
        converted_.clear();
        converted_.reserve(slice.n_events);
        for (const auto& event : *slice.events) {
            if (event.x >= width_ || event.y >= height_ || event.p > 1 || event.t < start_us ||
                event.t >= end_us || (have_last_event_ && event.t < last_event_us_))
                throw std::runtime_error("Invalid or out-of-order SDK CD event");
            converted_.push_back({static_cast<std::uint16_t>(event.x),
                                  static_cast<std::uint16_t>(event.y), event.t,
                                  static_cast<std::uint8_t>(event.p)});
            last_event_us_ = event.t;
            have_last_event_ = true;
        }
        peak_window_events_ = std::max(peak_window_events_, converted_.size());
        window = {converted_,
                  width_,
                  height_,
                  start_us,
                  end_us,
                  sequence_++,
                  end_us - start_us < window_us_};
        next_start_us_ = end_us;
        return true;
    }

    [[nodiscard]] std::size_t peak_window_events() const noexcept override {
        return peak_window_events_;
    }

  private:
    static Metavision::Camera open_camera(const std::filesystem::path& path) {
        if (path.extension() != ".hdf5")
            throw std::invalid_argument("Expected .hdf5 input");
        Metavision::FileConfigHints hints;
        hints.real_time_playback(false).time_shift(false);
        return Metavision::Camera::from_file(path, hints);
    }

    static std::int64_t checked_last_end(std::int64_t t, std::size_t count) {
        if (count == 0 || t == std::numeric_limits<std::int64_t>::max())
            throw std::runtime_error("Invalid final SDK slice timestamp");
        return t + 1;
    }

    std::int64_t first_start(const Metavision::Slice& slice, std::int64_t end_us, bool full) const {
        if (slice.events->empty())
            throw std::runtime_error("First SDK slice has no CD events");
        return full ? end_us - window_us_ : slice.events->front().t;
    }

    Metavision::CameraStreamSlicer slicer_;
    Metavision::CameraStreamSlicer::SliceIterator iterator_;
    std::int64_t window_us_;
    std::size_t max_window_events_;
    std::uint32_t width_, height_;
    std::vector<Event> converted_;
    std::size_t peak_window_events_ = 0;
    std::int64_t next_start_us_ = 0, last_event_us_ = 0;
    std::uint64_t sequence_ = 0;
    bool started_ = false, have_last_event_ = false;
};

} // namespace

std::unique_ptr<TimeSliceSource> open_metavision_hdf5(const std::filesystem::path& path,
                                                      std::int64_t window_us,
                                                      std::size_t max_window_events) {
    if (window_us <= 0 || max_window_events == 0)
        throw std::invalid_argument("Window duration and event cap must be positive");
    return std::make_unique<MetavisionHdf5TimeSliceSource>(path, window_us, max_window_events);
}

} // namespace event_voxel::data
