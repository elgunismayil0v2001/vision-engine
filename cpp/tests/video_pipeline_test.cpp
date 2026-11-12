#include "vision_engine/preprocess.hpp"
#include "vision_engine/video_pipeline_cuda.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace vision_engine {
namespace {

TEST(CudaVideoPipelineTest, ExplicitCpuFallbackMatchesScalarForAllFrames) {
    constexpr std::size_t width = 7;
    constexpr std::size_t height = 5;
    constexpr std::size_t frame_count = 3;
    constexpr std::size_t frame_elements = width * height * 3;
    std::vector<std::uint8_t> input(frame_count * frame_elements);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] = static_cast<std::uint8_t>((index * 29 + 17) % 256);
    }
    std::vector<float> actual(frame_count * frame_elements);

    CudaVideoPipeline pipeline(
        width, height, 2, VideoPipelineMode::cpu_fallback
    );
    const VideoPipelineStats stats = pipeline.process(input, frame_count, actual);

    EXPECT_FALSE(pipeline.using_cuda());
    EXPECT_FALSE(stats.cuda_active);
    EXPECT_EQ(stats.frame_count, frame_count);
    EXPECT_GT(stats.fps, 0.0);
    for (std::size_t frame = 0; frame < frame_count; ++frame) {
        std::vector<float> expected(frame_elements);
        normalize_rgb_to_nchw_scalar(
            std::span<const std::uint8_t>(input).subspan(frame * frame_elements, frame_elements),
            width,
            height,
            expected
        );
        const auto actual_frame = std::span<const float>(actual).subspan(
            frame * frame_elements, frame_elements
        );
        EXPECT_TRUE(std::equal(actual_frame.begin(), actual_frame.end(), expected.begin()));
    }
    EXPECT_GT(stats.cpu_fallback.p50_ms, 0.0);
}

TEST(CudaVideoPipelineTest, AutomaticModeMatchesScalarAndRejectsBadBuffers) {
    constexpr std::size_t width = 5;
    constexpr std::size_t height = 3;
    constexpr std::size_t frame_elements = width * height * 3;
    const std::vector<std::uint8_t> input(frame_elements * 2, 127);
    std::vector<float> actual(frame_elements * 2);

    CudaVideoPipeline pipeline(width, height, 2);
    const VideoPipelineStats stats = pipeline.process(input, 2, actual);

    EXPECT_EQ(stats.frame_count, 2U);
    EXPECT_EQ(stats.cuda_active, pipeline.using_cuda());
    for (std::size_t frame = 0; frame < 2; ++frame) {
        std::vector<float> expected(frame_elements);
        normalize_rgb_to_nchw_scalar(
            std::span<const std::uint8_t>(input).subspan(frame * frame_elements, frame_elements),
            width,
            height,
            expected
        );
        for (std::size_t index = 0; index < frame_elements; ++index) {
            EXPECT_NEAR(actual[frame * frame_elements + index], expected[index], 1e-6F);
        }
    }
    EXPECT_THROW(pipeline.process(input, 2, std::span<float>{}), std::invalid_argument);
}

}  // namespace
}  // namespace vision_engine
