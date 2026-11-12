#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace vision_engine::detail {

inline constexpr std::size_t kChannels = 3;

inline std::size_t checked_element_count(std::size_t width, std::size_t height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument("image width and height must be positive");
    }
    if (width > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error("image dimensions overflow");
    }
    const std::size_t pixels = width * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / kChannels) {
        throw std::overflow_error("image element count overflows");
    }
    return pixels * kChannels;
}

inline std::size_t validate_buffers(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
) {
    const std::size_t element_count = checked_element_count(width, height);
    if (rgb.size() != element_count) {
        throw std::invalid_argument("RGB input size does not match image dimensions");
    }
    if (output.size() != element_count) {
        throw std::invalid_argument("output size does not match image dimensions");
    }
    return element_count / kChannels;
}

}  // namespace vision_engine::detail
