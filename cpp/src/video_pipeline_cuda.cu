#include "vision_engine/video_pipeline_cuda.hpp"

#include "preprocess_common.hpp"
#include "preprocess_cuda_internal.hpp"
#include "vision_engine/preprocess.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision_engine {
namespace {

using Clock = std::chrono::steady_clock;

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

double elapsed_ms(Clock::time_point start, Clock::time_point finish) {
    return std::chrono::duration<double, std::milli>(finish - start).count();
}

double cuda_elapsed_ms(cudaEvent_t start, cudaEvent_t finish) {
    float elapsed = 0.0F;
    cuda_check(cudaEventElapsedTime(&elapsed, start, finish), "read CUDA stage timing");
    return static_cast<double>(elapsed);
}

double percentile(std::vector<double>& values, double percentile_value) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(percentile_value * static_cast<double>(values.size()));
    return values[static_cast<std::size_t>(rank) - 1];
}

StageLatencySummary summarize(std::vector<double>& values) {
    return {
        percentile(values, 0.50),
        percentile(values, 0.99),
        percentile(values, 0.999),
    };
}

struct TimingSamples {
    std::vector<double> host_input_copy;
    std::vector<double> host_to_device;
    std::vector<double> normalize_kernel;
    std::vector<double> device_to_host;
    std::vector<double> host_output_copy;
    std::vector<double> end_to_end;
    std::vector<double> cpu_fallback;

    void clear() {
        host_input_copy.clear();
        host_to_device.clear();
        normalize_kernel.clear();
        device_to_host.clear();
        host_output_copy.clear();
        end_to_end.clear();
        cpu_fallback.clear();
    }

    void reserve(std::size_t frame_count) {
        host_input_copy.reserve(frame_count);
        host_to_device.reserve(frame_count);
        normalize_kernel.reserve(frame_count);
        device_to_host.reserve(frame_count);
        host_output_copy.reserve(frame_count);
        end_to_end.reserve(frame_count);
        cpu_fallback.reserve(frame_count);
    }
};

struct StreamSlot {
    cudaStream_t stream = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_finish = nullptr;
    cudaEvent_t kernel_start = nullptr;
    cudaEvent_t kernel_finish = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_finish = nullptr;
    std::uint8_t* pinned_input = nullptr;
    float* pinned_output = nullptr;
    std::uint8_t* device_input = nullptr;
    float* device_output = nullptr;
    std::size_t frame_index = 0;
    Clock::time_point submitted_at{};
    double host_input_copy_ms = 0.0;
    bool pending = false;
};

bool create_slot(
    StreamSlot& slot,
    std::size_t input_bytes,
    std::size_t output_bytes
) noexcept {
    if (cudaStreamCreateWithFlags(&slot.stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&slot.pinned_input), input_bytes,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&slot.pinned_output), output_bytes,
                      cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&slot.device_input), input_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&slot.device_output), output_bytes) != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    const std::array<cudaEvent_t*, 6> events{
        &slot.h2d_start,
        &slot.h2d_finish,
        &slot.kernel_start,
        &slot.kernel_finish,
        &slot.d2h_start,
        &slot.d2h_finish,
    };
    for (cudaEvent_t* event : events) {
        if (cudaEventCreate(event) != cudaSuccess) {
            static_cast<void>(cudaGetLastError());
            return false;
        }
    }
    return true;
}

void release_slot(StreamSlot& slot) noexcept {
    if (slot.stream != nullptr) {
        static_cast<void>(cudaStreamSynchronize(slot.stream));
    }
    for (cudaEvent_t event : {slot.h2d_start, slot.h2d_finish,
                              slot.kernel_start, slot.kernel_finish,
                              slot.d2h_start, slot.d2h_finish}) {
        if (event != nullptr) {
            static_cast<void>(cudaEventDestroy(event));
        }
    }
    if (slot.device_output != nullptr) {
        static_cast<void>(cudaFree(slot.device_output));
    }
    if (slot.device_input != nullptr) {
        static_cast<void>(cudaFree(slot.device_input));
    }
    if (slot.pinned_output != nullptr) {
        static_cast<void>(cudaFreeHost(slot.pinned_output));
    }
    if (slot.pinned_input != nullptr) {
        static_cast<void>(cudaFreeHost(slot.pinned_input));
    }
    if (slot.stream != nullptr) {
        static_cast<void>(cudaStreamDestroy(slot.stream));
    }
    slot = {};
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
          pixel_count_(element_count_ / detail::kChannels),
          input_bytes_(element_count_),
          output_bytes_(checked_output_bytes(element_count_)) {
        if (stream_count == 0) {
            throw std::invalid_argument("video pipeline stream count must be positive");
        }
        if (mode == VideoPipelineMode::cpu_fallback) {
            return;
        }

        int device_count = 0;
        const cudaError_t query_result = cudaGetDeviceCount(&device_count);
        if (query_result != cudaSuccess || device_count == 0) {
            static_cast<void>(cudaGetLastError());
            return;
        }

        slots_.resize(stream_count);
        for (StreamSlot& slot : slots_) {
            if (!create_slot(slot, input_bytes_, output_bytes_)) {
                release_slots();
                return;
            }
        }
        using_cuda_ = true;
    }

    ~Impl() {
        release_slots();
    }

    VideoPipelineStats process(
        std::span<const std::uint8_t> rgb_frames,
        std::size_t frame_count,
        std::span<float> normalized_frames
    ) {
        validate_frame_buffers(rgb_frames, frame_count, normalized_frames);
        VideoPipelineStats result;
        result.frame_count = frame_count;
        result.cuda_active = using_cuda_;
        result.stream_count = using_cuda_ ? slots_.size() : 0;
        TimingSamples& samples = samples_;
        samples.clear();
        samples.reserve(frame_count);
        const Clock::time_point wall_start = Clock::now();

        if (!using_cuda_) {
            for (std::size_t frame = 0; frame < frame_count; ++frame) {
                const Clock::time_point frame_start = Clock::now();
                normalize_rgb_to_nchw_scalar(
                    rgb_frames.subspan(frame * input_bytes_, input_bytes_),
                    width_,
                    height_,
                    normalized_frames.subspan(frame * element_count_, element_count_)
                );
                const double frame_ms = elapsed_ms(frame_start, Clock::now());
                samples.cpu_fallback.push_back(frame_ms);
                samples.end_to_end.push_back(frame_ms);
            }
            finish_stats(result, samples, wall_start, frame_count);
            return result;
        }

        try {
            for (std::size_t frame = 0; frame < frame_count; ++frame) {
                StreamSlot& slot = slots_[frame % slots_.size()];
                if (slot.pending) {
                    complete_slot(slot, normalized_frames, samples);
                }
                slot.frame_index = frame;
                slot.submitted_at = Clock::now();
                const Clock::time_point input_copy_start = Clock::now();
                std::memcpy(
                    slot.pinned_input,
                    rgb_frames.data() + frame * input_bytes_,
                    input_bytes_
                );
                slot.host_input_copy_ms = elapsed_ms(input_copy_start, Clock::now());
                slot.pending = true;

                cuda_check(cudaEventRecord(slot.h2d_start, slot.stream), "record H2D start");
                cuda_check(
                    cudaMemcpyAsync(
                        slot.device_input,
                        slot.pinned_input,
                        input_bytes_,
                        cudaMemcpyHostToDevice,
                        slot.stream
                    ),
                    "queue RGB frame copy"
                );
                cuda_check(
                    cudaEventRecord(slot.h2d_finish, slot.stream), "record H2D finish"
                );
                cuda_check(
                    cudaEventRecord(slot.kernel_start, slot.stream), "record kernel start"
                );
                detail::launch_normalize_rgb_cuda(
                    slot.device_input, pixel_count_, slot.device_output, slot.stream
                );
                cuda_check(
                    cudaEventRecord(slot.kernel_finish, slot.stream), "record kernel finish"
                );
                cuda_check(cudaEventRecord(slot.d2h_start, slot.stream), "record D2H start");
                cuda_check(
                    cudaMemcpyAsync(
                        slot.pinned_output,
                        slot.device_output,
                        output_bytes_,
                        cudaMemcpyDeviceToHost,
                        slot.stream
                    ),
                    "queue normalized frame copy"
                );
                cuda_check(
                    cudaEventRecord(slot.d2h_finish, slot.stream), "record D2H finish"
                );
            }
            for (StreamSlot& slot : slots_) {
                if (slot.pending) {
                    complete_slot(slot, normalized_frames, samples);
                }
            }
        } catch (...) {
            for (StreamSlot& slot : slots_) {
                static_cast<void>(cudaStreamSynchronize(slot.stream));
                slot.pending = false;
            }
            static_cast<void>(cudaGetLastError());
            throw;
        }
        finish_stats(result, samples, wall_start, frame_count);
        return result;
    }

    [[nodiscard]] bool using_cuda() const noexcept {
        return using_cuda_;
    }

private:
    static std::size_t checked_output_bytes(std::size_t element_count) {
        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("video output byte count overflows");
        }
        return element_count * sizeof(float);
    }

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
        if (rgb_frames.size() != frame_count * input_bytes_) {
            throw std::invalid_argument("video RGB buffer size does not match frame count");
        }
        if (normalized_frames.size() != frame_count * element_count_) {
            throw std::invalid_argument("video output buffer size does not match frame count");
        }
    }

    void complete_slot(
        StreamSlot& slot,
        std::span<float> normalized_frames,
        TimingSamples& samples
    ) {
        cuda_check(cudaStreamSynchronize(slot.stream), "wait for stream slot completion");
        samples.host_input_copy.push_back(slot.host_input_copy_ms);
        samples.host_to_device.push_back(cuda_elapsed_ms(slot.h2d_start, slot.h2d_finish));
        samples.normalize_kernel.push_back(
            cuda_elapsed_ms(slot.kernel_start, slot.kernel_finish)
        );
        samples.device_to_host.push_back(cuda_elapsed_ms(slot.d2h_start, slot.d2h_finish));
        const Clock::time_point output_copy_start = Clock::now();
        std::memcpy(
            normalized_frames.data() + slot.frame_index * element_count_,
            slot.pinned_output,
            output_bytes_
        );
        samples.host_output_copy.push_back(elapsed_ms(output_copy_start, Clock::now()));
        samples.end_to_end.push_back(elapsed_ms(slot.submitted_at, Clock::now()));
        slot.pending = false;
    }

    void finish_stats(
        VideoPipelineStats& result,
        const TimingSamples& samples,
        Clock::time_point wall_start,
        std::size_t frame_count
    ) const {
        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - wall_start).count();
        result.host_input_copy = summarize(samples.host_input_copy);
        result.host_to_device = summarize(samples.host_to_device);
        result.normalize_kernel = summarize(samples.normalize_kernel);
        result.device_to_host = summarize(samples.device_to_host);
        result.host_output_copy = summarize(samples.host_output_copy);
        result.end_to_end = summarize(samples.end_to_end);
        result.cpu_fallback = summarize(samples.cpu_fallback);
        result.fps = static_cast<double>(frame_count) / wall_seconds;
    }

    void release_slots() noexcept {
        for (StreamSlot& slot : slots_) {
            release_slot(slot);
        }
        slots_.clear();
        using_cuda_ = false;
    }

    std::size_t width_;
    std::size_t height_;
    std::size_t element_count_;
    std::size_t pixel_count_;
    std::size_t input_bytes_;
    std::size_t output_bytes_;
    TimingSamples samples_;
    std::vector<StreamSlot> slots_;
    bool using_cuda_ = false;
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
