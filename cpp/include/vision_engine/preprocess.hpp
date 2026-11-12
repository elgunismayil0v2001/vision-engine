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

// Scalar implementation is the reference for every optimized path.
void normalize_rgb_to_nchw_scalar(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
);

// Uses a static OpenMP schedule when the project was built with OpenMP; otherwise
// it runs the scalar implementation. thread_count must always be positive.
void normalize_rgb_to_nchw_openmp(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output,
    int thread_count
);

// Uses ARM NEON where available and the scalar reference on other architectures.
void normalize_rgb_to_nchw_neon(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
);

[[nodiscard]] bool neon_available() noexcept;
[[nodiscard]] bool openmp_available() noexcept;

}  // namespace vision_engine
