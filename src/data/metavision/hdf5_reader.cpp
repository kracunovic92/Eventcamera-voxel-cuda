#include "event_voxel/data/metavision_hdf5.hpp"

#include <metavision/sdk/base/events/event_cd.h>
#include <metavision/sdk/stream/hdf5_event_file_reader.h>

#include <charconv>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <string>
#include <utility>
#ifdef EVENT_VOXEL_NVTX
#include <nvtx3/nvtx3.hpp>
#endif

namespace event_voxel::data {
namespace {

std::pair<std::uint32_t, std::uint32_t> parse_geometry(const std::string& geometry) {
    const auto separator = geometry.find('x');
    if (separator == std::string::npos)
        throw std::runtime_error("Missing HDF5 geometry WxH");
    std::uint32_t width = 0, height = 0;
    const auto [x_end, x_error] =
        std::from_chars(geometry.data(), geometry.data() + separator, width);
    const auto [y_end, y_error] =
        std::from_chars(geometry.data() + separator + 1, geometry.data() + geometry.size(), height);
    if (x_error != std::errc{} || y_error != std::errc{} || x_end != geometry.data() + separator ||
        y_end != geometry.data() + geometry.size() || width == 0 || height == 0 || width > 65536 ||
        height > 65536) {
        throw std::runtime_error("Invalid HDF5 sensor geometry");
    }
    return {width, height};
}

class MetavisionHdf5Reader final : public EventReader {
  public:
    MetavisionHdf5Reader(const std::filesystem::path& path, std::size_t cap)
        : reader_(path, false), cap_(cap) {
        if (path.extension() != ".hdf5" || cap_ == 0) {
            throw std::invalid_argument(
                "Expected .hdf5 input and positive decoded-block event cap");
        }
        const auto metadata = reader_.get_metadata_map();
        const auto geometry = metadata.find("geometry");
        if (geometry == metadata.end())
            throw std::runtime_error("HDF5 geometry metadata is absent");
        const auto [width, height] = parse_geometry(geometry->second);
        width_ = width;
        height_ = height;
        // Only the EventCD overload is registered: EXT_TRIGGER is never treated as CD.
        reader_.add_read_callback(
            Metavision::EventFileReader::EventsBufferReadCallback<Metavision::EventCD>{
                [this](const Metavision::EventCD* begin, const Metavision::EventCD* end) {
#ifdef EVENT_VOXEL_NVTX
                    nvtx3::scoped_range marker("convert CD events");
#endif
                    if (static_cast<std::size_t>(end - begin) > cap_) {
                        throw std::runtime_error("SDK CD block exceeds max_decoded_block_events");
                    }
                    for (auto it = begin; it != end; ++it) {
                        if (queue_.size() == cap_) {
                            throw std::runtime_error("SDK decoded CD queue exceeds cap");
                        }
                        // EventCD::t is in microseconds; the reader's time_shift=false
                        // preserves the recording's original timestamp origin and gaps.
                        queue_.push_back(Event{static_cast<std::uint16_t>(it->x),
                                               static_cast<std::uint16_t>(it->y), it->t,
                                               static_cast<std::uint8_t>(it->p)});
                    }
                }});
    }

    bool next(Event& event) override {
        while (queue_.empty()) {
#ifdef EVENT_VOXEL_NVTX
            nvtx3::scoped_range marker("HDF5 SDK decode");
#endif
            if (!reader_.read())
                return false;
        }
        event = queue_.front();
        queue_.pop_front();
        return true;
    }
    std::uint32_t width() const override {
        return width_;
    }
    std::uint32_t height() const override {
        return height_;
    }

  private:
    Metavision::HDF5EventFileReader reader_;
    std::size_t cap_;
    std::uint32_t width_ = 0;
    std::uint32_t height_ = 0;
    std::deque<Event> queue_;
};

} // namespace

std::unique_ptr<EventReader> open_metavision_hdf5(const std::filesystem::path& path,
                                                  std::size_t max_decoded_block_events) {
    return std::make_unique<MetavisionHdf5Reader>(path, max_decoded_block_events);
}

} // namespace event_voxel::data
