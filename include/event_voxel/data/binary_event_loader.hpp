
#pragma once

#include <filesystem>

#include "event_voxel/data/event_loader.hpp"

namespace event_voxel {

class BinaryEventLoader final : public IEventLoader {
  public:
    explicit BinaryEventLoader(std::filesystem::path path);

    [[nodiscard]]
    EventBatch load() override;

  private:
    std::filesystem::path path_;
};

} // namespace event_voxel
