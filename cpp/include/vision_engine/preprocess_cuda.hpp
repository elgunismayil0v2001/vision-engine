#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace vision_engine {

enum class CudaPreprocessorMode {
    automatic,
    cpu_fallback,
};

// Owns reusable device buffers. If no CUDA device is present (or device-buffer
// allocation fails), process() uses the scalar CPU implementation instead.
class CudaPreprocessor {
public:
    CudaPreprocessor(
        std::size_t width,
        std::size_t height,
        CudaPreprocessorMode mode = CudaPreprocessorMode::automatic
    );
    ~CudaPreprocessor() noexcept;

    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;
    CudaPreprocessor(CudaPreprocessor&&) = delete;
    CudaPreprocessor& operator=(CudaPreprocessor&&) = delete;

    void process(
        std::span<const std::uint8_t> rgb,
        std::span<float> output
    );

    [[nodiscard]] bool using_cuda() const noexcept;
    [[nodiscard]] static bool cuda_device_available() noexcept;
    [[nodiscard]] static std::string cuda_device_name();

private:
    std::size_t width_;
    std::size_t height_;
    std::size_t element_count_;
    std::size_t pixel_count_;
    std::uint8_t* device_rgb_ = nullptr;
    float* device_output_ = nullptr;
    bool using_cuda_ = false;
};

}  // namespace vision_engine
