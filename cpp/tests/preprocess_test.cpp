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

TEST(PreprocessTest, OptimizedCpuPathsMatchScalarReference) {
    constexpr std::size_t width = 19;
    constexpr std::size_t height = 7;
    constexpr std::size_t element_count = width * height * 3;
    std::vector<std::uint8_t> rgb(element_count);
    for (std::size_t index = 0; index < rgb.size(); ++index) {
        rgb[index] = static_cast<std::uint8_t>((index * 37 + 13) % 256);
    }
    std::array<std::vector<float>, 3> outputs{
        std::vector<float>(element_count),
        std::vector<float>(element_count),
        std::vector<float>(element_count),
    };

    normalize_rgb_to_nchw_scalar(rgb, width, height, outputs[0]);
    normalize_rgb_to_nchw_openmp(rgb, width, height, outputs[1], 3);
    normalize_rgb_to_nchw_neon(rgb, width, height, outputs[2]);

    for (std::size_t variant = 1; variant < outputs.size(); ++variant) {
        for (std::size_t index = 0; index < element_count; ++index) {
            EXPECT_NEAR(outputs[variant][index], outputs[0][index], 1e-6F)
                << "variant=" << variant << " index=" << index;
        }
    }
}

TEST(PreprocessTest, OpenMpRejectsAnInvalidThreadCount) {
    const std::array<std::uint8_t, 3> rgb{0, 0, 0};
    std::array<float, 3> output{};

    EXPECT_THROW(
        normalize_rgb_to_nchw_openmp(rgb, 1, 1, output, 0), std::invalid_argument
    );
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
