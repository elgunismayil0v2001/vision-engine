#include "vision_engine/preprocess_cuda.hpp"

#include "preprocess_common.hpp"
#include "vision_engine/preprocess.hpp"

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace vision_engine {
namespace {

constexpr unsigned int kThreadsPerBlock = 256;
__device__ __constant__ float kMean[3]{0.485F, 0.456F, 0.406F};
__device__ __constant__ float kStd[3]{0.229F, 0.224F, 0.225F};

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(result)
        );
    }
}

__global__ void normalize_rgb_kernel(
    const std::uint8_t* rgb,
    std::size_t pixel_count,
    float* output
) {
    const std::size_t pixel =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixel_count) {
        return;
    }
    const std::size_t source = pixel * detail::kChannels;
    for (std::size_t channel = 0; channel < detail::kChannels; ++channel) {
        const float unit_value = static_cast<float>(rgb[source + channel]) / 255.0F;
        output[channel * pixel_count + pixel] =
            (unit_value - kMean[channel]) / kStd[channel];
    }
}

}  // namespace

CudaPreprocessor::CudaPreprocessor(
    std::size_t width,
    std::size_t height,
    CudaPreprocessorMode mode
)
    : width_(width),
      height_(height),
      element_count_(detail::checked_element_count(width, height)),
      pixel_count_(element_count_ / detail::kChannels) {
    if (element_count_ > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("normalized output allocation size overflows");
    }
    if (mode == CudaPreprocessorMode::cpu_fallback) {
        return;
    }
    int device_count = 0;
    const cudaError_t query_result = cudaGetDeviceCount(&device_count);
    if (query_result != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        return;
    }

    const cudaError_t input_result =
        cudaMalloc(reinterpret_cast<void**>(&device_rgb_), element_count_);
    if (input_result != cudaSuccess) {
        device_rgb_ = nullptr;
        static_cast<void>(cudaGetLastError());
        return;
    }
    const cudaError_t output_result =
        cudaMalloc(reinterpret_cast<void**>(&device_output_), element_count_ * sizeof(float));
    if (output_result != cudaSuccess) {
        static_cast<void>(cudaFree(device_rgb_));
        device_rgb_ = nullptr;
        device_output_ = nullptr;
        static_cast<void>(cudaGetLastError());
        return;
    }
    using_cuda_ = true;
}

CudaPreprocessor::~CudaPreprocessor() noexcept {
    if (device_output_ != nullptr) {
        static_cast<void>(cudaFree(device_output_));
    }
    if (device_rgb_ != nullptr) {
        static_cast<void>(cudaFree(device_rgb_));
    }
}

void CudaPreprocessor::process(
    std::span<const std::uint8_t> rgb,
    std::span<float> output
) {
    const std::size_t actual_pixels =
        detail::validate_buffers(rgb, width_, height_, output);
    if (actual_pixels != pixel_count_ || output.size() != element_count_) {
        throw std::invalid_argument("buffers do not match this CUDA preprocessor");
    }
    if (!using_cuda_) {
        normalize_rgb_to_nchw_scalar(rgb, width_, height_, output);
        return;
    }

    cuda_check(
        cudaMemcpy(device_rgb_, rgb.data(), element_count_, cudaMemcpyHostToDevice),
        "copy RGB input to CUDA device"
    );
    const std::size_t block_count =
        (pixel_count_ + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (block_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("image is too large for the CUDA grid dimensions");
    }
    normalize_rgb_kernel<<<static_cast<unsigned int>(block_count), kThreadsPerBlock>>>(
        device_rgb_, pixel_count_, device_output_
    );
    cuda_check(cudaGetLastError(), "launch CUDA normalization kernel");
    cuda_check(
        cudaMemcpy(
            output.data(),
            device_output_,
            element_count_ * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy normalized output from CUDA device"
    );
}

bool CudaPreprocessor::using_cuda() const noexcept {
    return using_cuda_;
}

bool CudaPreprocessor::cuda_device_available() noexcept {
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess) {
        static_cast<void>(cudaGetLastError());
        return false;
    }
    return device_count > 0;
}

std::string CudaPreprocessor::cuda_device_name() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get current CUDA device");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "query CUDA device");
    return properties.name;
}

}  // namespace vision_engine
