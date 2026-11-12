#include "vision_engine/preprocess.hpp"

#include "preprocess_common.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

namespace vision_engine {
namespace {

constexpr std::array<float, 3> kMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3> kStd{0.229F, 0.224F, 0.225F};

void normalize_scalar_validated(
    std::span<const std::uint8_t> rgb,
    std::size_t pixel_count,
    std::span<float> output
) {
    for (std::size_t pixel = 0; pixel < pixel_count; ++pixel) {
        const std::size_t source = pixel * detail::kChannels;
        for (std::size_t channel = 0; channel < detail::kChannels; ++channel) {
            const float unit_value = static_cast<float>(rgb[source + channel]) / 255.0F;
            output[channel * pixel_count + pixel] =
                (unit_value - kMean[channel]) / kStd[channel];
        }
    }
}

}  // namespace

void normalize_rgb_to_nchw(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
) {
    normalize_rgb_to_nchw_scalar(rgb, width, height, output);
}

void normalize_rgb_to_nchw_scalar(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
) {
    const std::size_t pixel_count = detail::validate_buffers(rgb, width, height, output);
    normalize_scalar_validated(rgb, pixel_count, output);
}

void normalize_rgb_to_nchw_openmp(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output,
    int thread_count
) {
    const std::size_t pixel_count = detail::validate_buffers(rgb, width, height, output);
    if (thread_count < 1) {
        throw std::invalid_argument("OpenMP thread count must be positive");
    }
#if defined(VISION_ENGINE_HAS_OPENMP)
    if (pixel_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("image is too large for the OpenMP loop index");
    }
    const auto signed_pixel_count = static_cast<std::int64_t>(pixel_count);
#pragma omp parallel for schedule(static) num_threads(thread_count)
    for (std::int64_t signed_pixel = 0; signed_pixel < signed_pixel_count; ++signed_pixel) {
        const std::size_t pixel = static_cast<std::size_t>(signed_pixel);
        const std::size_t source = pixel * detail::kChannels;
        for (std::size_t channel = 0; channel < detail::kChannels; ++channel) {
            const float unit_value = static_cast<float>(rgb[source + channel]) / 255.0F;
            output[channel * pixel_count + pixel] =
                (unit_value - kMean[channel]) / kStd[channel];
        }
    }
#else
    normalize_scalar_validated(rgb, pixel_count, output);
#endif
}

void normalize_rgb_to_nchw_neon(
    std::span<const std::uint8_t> rgb,
    std::size_t width,
    std::size_t height,
    std::span<float> output
) {
    const std::size_t pixel_count = detail::validate_buffers(rgb, width, height, output);
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    const std::size_t vectorized_count = pixel_count - (pixel_count % 8);
    const float32x4_t reciprocal_255 = vdupq_n_f32(1.0F / 255.0F);
    const std::array<float32x4_t, 3> means{
        vdupq_n_f32(kMean[0]), vdupq_n_f32(kMean[1]), vdupq_n_f32(kMean[2])};
    const std::array<float32x4_t, 3> reciprocal_stds{
        vdupq_n_f32(1.0F / kStd[0]),
        vdupq_n_f32(1.0F / kStd[1]),
        vdupq_n_f32(1.0F / kStd[2]),
    };

    for (std::size_t pixel = 0; pixel < vectorized_count; pixel += 8) {
        const uint8x8x3_t channels = vld3_u8(rgb.data() + pixel * detail::kChannels);
        for (std::size_t channel = 0; channel < detail::kChannels; ++channel) {
            const uint16x8_t values_16 = vmovl_u8(channels.val[channel]);
            const uint32x4_t values_low = vmovl_u16(vget_low_u16(values_16));
            const uint32x4_t values_high = vmovl_u16(vget_high_u16(values_16));
            const float32x4_t unit_low = vmulq_f32(vcvtq_f32_u32(values_low), reciprocal_255);
            const float32x4_t unit_high =
                vmulq_f32(vcvtq_f32_u32(values_high), reciprocal_255);
            const float32x4_t normalized_low =
                vmulq_f32(vsubq_f32(unit_low, means[channel]), reciprocal_stds[channel]);
            const float32x4_t normalized_high =
                vmulq_f32(vsubq_f32(unit_high, means[channel]), reciprocal_stds[channel]);
            vst1q_f32(output.data() + channel * pixel_count + pixel, normalized_low);
            vst1q_f32(output.data() + channel * pixel_count + pixel + 4, normalized_high);
        }
    }
    for (std::size_t pixel = vectorized_count; pixel < pixel_count; ++pixel) {
        const std::size_t source = pixel * detail::kChannels;
        for (std::size_t channel = 0; channel < detail::kChannels; ++channel) {
            const float unit_value = static_cast<float>(rgb[source + channel]) / 255.0F;
            output[channel * pixel_count + pixel] =
                (unit_value - kMean[channel]) / kStd[channel];
        }
    }
#else
    normalize_scalar_validated(rgb, pixel_count, output);
#endif
}

bool neon_available() noexcept {
#if (defined(__ARM_NEON) || defined(__ARM_NEON__)) && defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

bool openmp_available() noexcept {
#if defined(VISION_ENGINE_HAS_OPENMP)
    return true;
#else
    return false;
#endif
}

}  // namespace vision_engine
