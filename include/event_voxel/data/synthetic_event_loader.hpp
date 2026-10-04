
#pragma once

#include <utility>

#include "event_voxel/data/event_loader.hpp"
#include "event_voxel/synthetic_generator.hpp"

namespace event_voxel {

class SyntheticEventLoader final : public IEventLoader {
  public:
    explicit SyntheticEventLoader(GeneratorConfig config) : config_(std::move(config)) {}

    [[nodiscard]]
    EventBatch load() override {
        return {generate_events(config_), config_.width, config_.height, 0, config_.duration_us};
    }

  private:
    GeneratorConfig config_;
};

} // namespace event_voxel
