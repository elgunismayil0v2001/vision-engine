#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace vision_engine {

enum class VideoPipelineMode {
    automatic,
    cpu_fallback,
};

struct StageLatencySummary {
    double p50_ms = 0.0;
    double p99_ms = 0.0;
    double p999_ms = 0.0;
};

struct VideoPipelineStats {
    std::size_t frame_count = 0;
    double fps = 0.0;
    bool cuda_active = false;
    std::size_t stream_count = 0;
    StageLatencySummary host_input_copy;
    StageLatencySummary host_to_device;
    StageLatencySummary normalize_kernel;
    StageLatencySummary device_to_host;
    StageLatencySummary host_output_copy;
    StageLatencySummary end_to_end;
    StageLatencySummary cpu_fallback;
};

// Processes a contiguous sequence of already-decoded RGB frames. CUDA mode
// double-buffers input/output in pinned memory and runs each slot on its own
// stream. When CUDA is unavailable, process() normalizes on the CPU.
class CudaVideoPipeline {
public:
    CudaVideoPipeline(
        std::size_t width,
        std::size_t height,
        std::size_t stream_count,
        VideoPipelineMode mode = VideoPipelineMode::automatic
    );
    ~CudaVideoPipeline();

    CudaVideoPipeline(const CudaVideoPipeline&) = delete;
    CudaVideoPipeline& operator=(const CudaVideoPipeline&) = delete;
    CudaVideoPipeline(CudaVideoPipeline&&) noexcept;
    CudaVideoPipeline& operator=(CudaVideoPipeline&&) noexcept;

    VideoPipelineStats process(
        std::span<const std::uint8_t> rgb_frames,
        std::size_t frame_count,
        std::span<float> normalized_frames
    );

    [[nodiscard]] bool using_cuda() const noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision_engine
