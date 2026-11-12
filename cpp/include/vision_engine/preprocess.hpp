#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace vision_engine {

// Converts interleaved RGB uint8 pixels to ImageNet-normalized planar NCHW data.
// The input buffer is one image and the output buffer must hold 3 * width * height floats.
void normalize_rgb_to_nchw(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
);

}  // namespace vision_engine
