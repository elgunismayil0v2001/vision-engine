#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace vision_engine::detail {

void launch_normalize_rgb_cuda(
    const std::uint8_t* device_rgb,
    std::size_t pixel_count,
    float* device_output,
    cudaStream_t stream
);

}  // namespace vision_engine::detail
