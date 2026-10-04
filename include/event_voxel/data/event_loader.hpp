
#pragma once

#include "event_voxel/event_batch.hpp"

namespace event_voxel {

class IEventLoader {
  public:
    virtual ~IEventLoader() = default;

    [[nodiscard]]
    virtual EventBatch load() = 0;
};

} // namespace event_voxel
