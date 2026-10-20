#include "event_voxel/data/event_window_source.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace event_voxel::data {

EventWindowSource::EventWindowSource(std::unique_ptr<EventReader> reader, WindowConfig config)
    : reader_(std::move(reader)), config_(config) {
    if (!reader_ || reader_->width() == 0 || reader_->height() == 0 || config_.window_us <= 0 ||
        config_.chunk_us <= 0 || config_.max_chunk_events == 0 || config_.max_window_events == 0) {
        throw std::invalid_argument("Invalid event reader or window configuration");
    }
}

bool EventWindowSource::refill() {
    chunk_.clear();
    position_ = 0;
    if (eof_)
        return false;
    Event event{};
    while (chunk_.size() < config_.max_chunk_events) {
        if (pending_) {
            event = *pending_;
            pending_.reset();
        } else if (!reader_->next(event)) {
            eof_ = true;
            break;
        }
        if ((chunk_.empty() && started_ && event.timestamp_us < last_us_) ||
            (!chunk_.empty() && event.timestamp_us < chunk_.back().timestamp_us)) {
            throw std::runtime_error("CD timestamps are not ordered");
        }
        if (event.x >= reader_->width() || event.y >= reader_->height() || event.polarity > 1 ||
            event.timestamp_us < 0) {
            throw std::runtime_error("Invalid decoded CD event");
        }
        if (!chunk_.empty() &&
            event.timestamp_us - chunk_.front().timestamp_us >= config_.chunk_us) {
            pending_ = event;
            break;
        }
        chunk_.push_back(event);
        last_us_ = event.timestamp_us;
    }
    peak_buffered_events_ =
        std::max(peak_buffered_events_, chunk_.size() + staging_.size() + pending_.has_value());
    return !chunk_.empty();
}

bool EventWindowSource::peek(Event& event) {
    if (position_ == chunk_.size() && !refill())
        return false;
    event = chunk_[position_];
    return true;
}

bool EventWindowSource::next(EventWindowView& window) {
    staging_.clear();
    Event event{};
    if (!started_) {
        if (!peek(event))
            return false;
        started_ = true;
        start_us_ = event.timestamp_us;
    } else if (!peek(event) && start_us_ > last_us_) {
        return false;
    }
    if (start_us_ > std::numeric_limits<std::int64_t>::max() - config_.window_us) {
        throw std::overflow_error("Window timestamp overflow");
    }
    const auto nominal_end = start_us_ + config_.window_us;
    std::size_t begin = position_;
    std::size_t count = 0;
    while (true) {
        if (position_ == chunk_.size()) {
            if (begin < position_) {
                staging_.insert(staging_.end(), chunk_.begin() + static_cast<std::ptrdiff_t>(begin),
                                chunk_.end());
            }
            begin = 0;
            if (!refill())
                break;
        }
        event = chunk_[position_];
        if (event.timestamp_us >= nominal_end)
            break;
        if (count == config_.max_window_events) {
            throw std::runtime_error("A single window exceeds max_window_events; raise the cap");
        }
        ++position_;
        ++count;
    }
    // If the chunk was exhausted, staging contains only boundary-spanning data.
    if (!staging_.empty() && begin < position_) {
        staging_.insert(staging_.end(), chunk_.begin() + static_cast<std::ptrdiff_t>(begin),
                        chunk_.begin() + static_cast<std::ptrdiff_t>(position_));
    }
    const bool partial =
        eof_ && !pending_ && position_ == chunk_.size() && last_us_ < nominal_end - 1;
    const auto actual_end = partial ? last_us_ + 1 : nominal_end;
    if (actual_end <= start_us_)
        return false;
    window = {count == 0         ? std::span<const Event>{}
              : staging_.empty() ? std::span<const Event>(chunk_.data() + begin, count)
                                 : std::span<const Event>(staging_),
              reader_->width(),
              reader_->height(),
              start_us_,
              actual_end,
              sequence_++,
              partial};
    start_us_ = nominal_end;
    return true;
}

} // namespace event_voxel::data
