#include "vision_engine/preprocess.hpp"
#include "vision_engine/preprocess_cuda.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vision_engine {
namespace {

TEST(CudaPreprocessTest, MatchesScalarAndFallsBackWithoutADevice) {
    constexpr std::size_t width = 37;
    constexpr std::size_t height = 19;
    constexpr std::size_t element_count = width * height * 3;
    std::vector<std::uint8_t> rgb(element_count);
    for (std::size_t index = 0; index < rgb.size(); ++index) {
        rgb[index] = static_cast<std::uint8_t>((index * 19 + 7) % 256);
    }
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    normalize_rgb_to_nchw_scalar(rgb, width, height, expected);

    CudaPreprocessor preprocessor(width, height);
    preprocessor.process(rgb, actual);

    for (std::size_t index = 0; index < element_count; ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-6F) << "index=" << index;
    }
    if (CudaPreprocessor::cuda_device_available()) {
        EXPECT_TRUE(preprocessor.using_cuda());
    } else {
        EXPECT_FALSE(preprocessor.using_cuda());
    }
}

TEST(CudaPreprocessTest, ExplicitCpuFallbackMatchesScalarWithOrWithoutGpu) {
    constexpr std::size_t width = 7;
    constexpr std::size_t height = 5;
    constexpr std::size_t element_count = width * height * 3;
    const std::vector<std::uint8_t> rgb(element_count, 127);
    std::vector<float> expected(element_count);
    std::vector<float> actual(element_count);
    normalize_rgb_to_nchw_scalar(rgb, width, height, expected);

    CudaPreprocessor preprocessor(
        width, height, CudaPreprocessorMode::cpu_fallback
    );
    EXPECT_FALSE(preprocessor.using_cuda());
    preprocessor.process(rgb, actual);

    EXPECT_EQ(actual, expected);
}

TEST(CudaPreprocessTest, ReusesItsBuffersAndRejectsWrongShape) {
    CudaPreprocessor preprocessor(2, 1);
    const std::vector<std::uint8_t> rgb{0, 1, 2, 3, 4, 5};
    std::vector<float> output(6);
    preprocessor.process(rgb, output);
    preprocessor.process(rgb, output);

    EXPECT_THROW(
        preprocessor.process(std::span<const std::uint8_t>{}, output),
        std::invalid_argument
    );
}

}  // namespace
}  // namespace vision_engine
