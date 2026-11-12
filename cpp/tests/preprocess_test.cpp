#include "vision_engine/preprocess.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vision_engine {
namespace {

TEST(PreprocessTest, ConvertsRgbPixelsToImageNetNchw) {
    const std::array<std::uint8_t, 6> rgb{0, 127, 255, 255, 64, 32};
    std::array<float, 6> output{};

    normalize_rgb_to_nchw(rgb, 2, 1, output);

    EXPECT_NEAR(output[0], (0.0F - 0.485F) / 0.229F, 1e-6F);
    EXPECT_NEAR(output[1], (1.0F - 0.485F) / 0.229F, 1e-6F);
    EXPECT_NEAR(output[2], (127.0F / 255.0F - 0.456F) / 0.224F, 1e-6F);
    EXPECT_NEAR(output[3], (64.0F / 255.0F - 0.456F) / 0.224F, 1e-6F);
    EXPECT_NEAR(output[4], (255.0F / 255.0F - 0.406F) / 0.225F, 1e-6F);
    EXPECT_NEAR(output[5], (32.0F / 255.0F - 0.406F) / 0.225F, 1e-6F);
}

TEST(PreprocessTest, RejectsInvalidDimensionsAndBufferSizes) {
    const std::array<std::uint8_t, 3> rgb{0, 0, 0};
    std::array<float, 3> output{};

    EXPECT_THROW(normalize_rgb_to_nchw(rgb, 0, 1, output), std::invalid_argument);
    EXPECT_THROW(normalize_rgb_to_nchw(rgb, 2, 1, output), std::invalid_argument);
    EXPECT_THROW(normalize_rgb_to_nchw(rgb, 1, 1, std::span<float>{}), std::invalid_argument);
}

}  // namespace
}  // namespace vision_engine
