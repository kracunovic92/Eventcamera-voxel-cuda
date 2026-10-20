#include "event_voxel/data/event_window_source.hpp"
#include "event_voxel/data/metavision_hdf5.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    constexpr std::size_t chunk_cap = 65'536;
    constexpr std::size_t window_cap = 1'000'000;
    event_voxel::data::EventWindowSource source(
        event_voxel::data::open_metavision_hdf5(std::filesystem::path(argv[1]), chunk_cap),
        {.window_us = 50'000,
         .chunk_us = 10'000'000,
         .max_chunk_events = chunk_cap,
         .max_window_events = window_cap});
    event_voxel::EventWindowView window;
    std::size_t total_events = 0, windows = 0;
    std::int64_t previous_end = -1;
    while (source.next(window)) {
        if (previous_end >= 0 && window.start_us != previous_end)
            throw std::runtime_error("Gap or overlap between windows");
        previous_end = window.end_us;
        for (const auto& event : window.events) {
            if (event.timestamp_us < window.start_us || event.timestamp_us >= window.end_us)
                throw std::runtime_error("CD event outside its window");
        }
        total_events += window.events.size();
        ++windows;
    }
    if (windows < 2 || total_events <= source.peak_buffered_events() ||
        source.peak_buffered_events() > chunk_cap + window_cap + 1)
        throw std::runtime_error("Streaming bound or recording size check failed");
    std::cout << "windows=" << windows << " events=" << total_events
              << " peak_buffered_events=" << source.peak_buffered_events() << '\n';
}
