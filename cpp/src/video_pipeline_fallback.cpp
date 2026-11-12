#if !defined(VISION_ENGINE_HAS_CUDA)

#include "vision_engine/video_pipeline_cuda.hpp"

#include "preprocess_common.hpp"
#include "vision_engine/preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vision_engine {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

StageLatencySummary summarize(std::vector<double>& values) {
    if (values.empty()) {
        return {};
    }
    std::sort(values.begin(), values.end());
    const auto percentile = [&values](double fraction) {
        const std::size_t rank = static_cast<std::size_t>(
            std::ceil(fraction * static_cast<double>(values.size()))
        );
        return values[rank - 1];
    };
    return {
        percentile(0.50),
        percentile(0.99),
        percentile(0.999),
    };
}

}  // namespace

class CudaVideoPipeline::Impl {
public:
    Impl(
        std::size_t width,
        std::size_t height,
        std::size_t stream_count,
        VideoPipelineMode mode
    )
        : width_(width),
          height_(height),
          element_count_(detail::checked_element_count(width, height)),
          input_bytes_(element_count_) {
        static_cast<void>(mode);
        if (stream_count == 0) {
            throw std::invalid_argument("video pipeline stream count must be positive");
        }
        if (element_count_ > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("video output byte count overflows");
        }
    }

    VideoPipelineStats process(
        std::span<const std::uint8_t> rgb_frames,
        std::size_t frame_count,
        std::span<float> normalized_frames
    ) {
        validate_frame_buffers(rgb_frames, frame_count, normalized_frames);
        frame_timings_.clear();
        frame_timings_.reserve(frame_count);

        VideoPipelineStats result;
        result.frame_count = frame_count;
        const Clock::time_point wall_start = Clock::now();
        for (std::size_t frame = 0; frame < frame_count; ++frame) {
            const Clock::time_point frame_start = Clock::now();
            normalize_rgb_to_nchw_scalar(
                rgb_frames.subspan(frame * input_bytes_, input_bytes_),
                width_,
                height_,
                normalized_frames.subspan(frame * element_count_, element_count_)
            );
            frame_timings_.push_back(elapsed_ms(frame_start, Clock::now()));
        }
        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - wall_start).count();
        result.fps = static_cast<double>(frame_count) / wall_seconds;
        result.cpu_fallback = summarize(frame_timings_);
        result.end_to_end = result.cpu_fallback;
        return result;
    }

    [[nodiscard]] bool using_cuda() const noexcept {
        return false;
    }

private:
    void validate_frame_buffers(
        std::span<const std::uint8_t> rgb_frames,
        std::size_t frame_count,
        std::span<float> normalized_frames
    ) const {
        if (frame_count == 0) {
            throw std::invalid_argument("video pipeline requires at least one frame");
        }
        if (frame_count > std::numeric_limits<std::size_t>::max() / input_bytes_ ||
            frame_count > std::numeric_limits<std::size_t>::max() / element_count_ ||
            frame_count > std::numeric_limits<std::size_t>::max() /
                              element_count_ / sizeof(float)) {
            throw std::overflow_error("video frame buffer size overflows");
        }
        const std::size_t total_elements = frame_count * element_count_;
        if (rgb_frames.size() != frame_count * input_bytes_) {
            throw std::invalid_argument("video RGB buffer size does not match frame count");
        }
        if (normalized_frames.size() != total_elements) {
            throw std::invalid_argument("video output buffer size does not match frame count");
        }
    }

    std::size_t width_;
    std::size_t height_;
    std::size_t element_count_;
    std::size_t input_bytes_;
    std::vector<double> frame_timings_;
};

CudaVideoPipeline::CudaVideoPipeline(
    std::size_t width,
    std::size_t height,
    std::size_t stream_count,
    VideoPipelineMode mode
)
    : impl_(std::make_unique<Impl>(width, height, stream_count, mode)) {}

CudaVideoPipeline::~CudaVideoPipeline() = default;
CudaVideoPipeline::CudaVideoPipeline(CudaVideoPipeline&&) noexcept = default;
CudaVideoPipeline& CudaVideoPipeline::operator=(CudaVideoPipeline&&) noexcept = default;

VideoPipelineStats CudaVideoPipeline::process(
    std::span<const std::uint8_t> rgb_frames,
    std::size_t frame_count,
    std::span<float> normalized_frames
) {
    if (impl_ == nullptr) {
        throw std::logic_error("video pipeline was moved from");
    }
    return impl_->process(rgb_frames, frame_count, normalized_frames);
}

bool CudaVideoPipeline::using_cuda() const noexcept {
    return impl_ != nullptr && impl_->using_cuda();
}

}  // namespace vision_engine

#endif  // !defined(VISION_ENGINE_HAS_CUDA)
