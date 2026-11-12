#include "vision_engine/preprocess.hpp"

#include <array>
#include <limits>
#include <stdexcept>

namespace vision_engine {
namespace {

constexpr std::array<float, 3> kMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kStd{0.229F, 0.224F, 0.225F};
constexpr std::size_t kChannels = 3;

std::size_t checked_element_count(std::size_t width, std::size_t height) {
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

}  // namespace

void normalize_rgb_to_nchw(
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

    const std::size_t pixels = element_count / kChannels;
    for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
        const std::size_t source = pixel * kChannels;
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
            const float unit_value = static_cast<float>(rgb[source + channel]) / 255.0F;
            output[channel * pixels + pixel] = (unit_value - kMean[channel]) / kStd[channel];
        }
    }
}

}  // namespace vision_engine
