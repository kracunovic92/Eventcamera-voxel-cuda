#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/resource.h>

namespace event_voxel::io {

struct RunRecord {
    std::string_view backend;
    std::string_view strategy;
    std::filesystem::path input;
    std::filesystem::path output;
    std::size_t source_events;
    std::size_t selected_events;
    std::size_t bins;
    std::size_t width;
    std::size_t height;
    std::size_t output_bytes;
    std::int64_t start_us;
    std::int64_t end_us;
    double load_ms;
    double voxelize_ms;
    double export_ms;
    std::optional<std::size_t> gpu_total_bytes;
    std::optional<std::size_t> gpu_free_before_bytes;
    std::optional<std::size_t> gpu_free_after_bytes;
};

inline long peak_rss_kib() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        throw std::runtime_error("getrusage failed");
    }
    return usage.ru_maxrss;
}

inline std::string json_string(std::string_view value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (const unsigned char character : value) {
        if (character == '"' || character == '\\') {
            result += '\\';
            result += static_cast<char>(character);
        } else if (character < 0x20) {
            result += "\\u00";
            result += hex[character >> 4];
            result += hex[character & 15];
        } else {
            result += static_cast<char>(character);
        }
    }
    return result + '"';
}

inline void write_run_record(const std::filesystem::path& path, const RunRecord& record) {
    if (path.empty()) {
        throw std::invalid_argument("Run record path cannot be empty");
    }
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open run record: " + path.string());
    }
    const auto number_or_null = [&](std::optional<std::size_t> value) {
        return value ? std::to_string(*value) : std::string("null");
    };
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"backend\": " << json_string(record.backend) << ",\n"
           << "  \"strategy\": " << json_string(record.strategy) << ",\n"
           << "  \"input\": " << json_string(record.input.string()) << ",\n"
           << "  \"output\": " << json_string(record.output.string()) << ",\n"
           << "  \"source_events\": " << record.source_events << ",\n"
           << "  \"selected_events\": " << record.selected_events << ",\n"
           << "  \"bins\": " << record.bins << ",\n"
           << "  \"width\": " << record.width << ",\n"
           << "  \"height\": " << record.height << ",\n"
           << "  \"start_us\": " << record.start_us << ",\n"
           << "  \"end_us\": " << record.end_us << ",\n"
           << "  \"output_bytes\": " << record.output_bytes << ",\n"
           << "  \"load_ms\": " << record.load_ms << ",\n"
           << "  \"voxelize_ms\": " << record.voxelize_ms << ",\n"
           << "  \"export_ms\": " << record.export_ms << ",\n"
           << "  \"peak_process_rss_kib\": " << peak_rss_kib() << ",\n"
           << "  \"gpu_total_bytes\": " << number_or_null(record.gpu_total_bytes) << ",\n"
           << "  \"gpu_free_before_bytes\": " << number_or_null(record.gpu_free_before_bytes)
           << ",\n"
           << "  \"gpu_free_after_bytes\": " << number_or_null(record.gpu_free_after_bytes) << "\n"
           << "}\n";
    if (!output) {
        throw std::runtime_error("Cannot write run record: " + path.string());
    }
}

} // namespace event_voxel::io
