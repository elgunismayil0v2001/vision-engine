#include "vision_engine/preprocess.hpp"
#include "vision_engine/preprocess_cuda.hpp"
#include "vision_engine/video_pipeline_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::size_t width = 640;
    std::size_t height = 480;
    std::size_t frames = 120;
    std::size_t warmup_frames = 16;
};

std::size_t parse_positive(const std::string& text, const char* name) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    std::size_t parsed_characters = 0;
    const unsigned long long parsed = std::stoull(text, &parsed_characters);
    if (parsed_characters != text.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string(name) + " must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--width") {
            options.width = parse_positive(value, "width");
        } else if (argument == "--height") {
            options.height = parse_positive(value, "height");
        } else if (argument == "--frames") {
            options.frames = parse_positive(value, "frames");
        } else if (argument == "--warmup-frames") {
            options.warmup_frames = parse_positive(value, "warmup frames");
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    return options;
}

std::string architecture_name() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

std::string operating_system_name() {
#if defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "unknown";
#endif
}

double max_abs_error(std::span<const float> lhs, std::span<const float> rhs) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(lhs[index]) - rhs[index])
        );
    }
    return maximum;
}

void append_stage(std::string& json, const vision_engine::StageLatencySummary& stage) {
    json += "{\"p50_ms\":" + std::to_string(stage.p50_ms);
    json += ",\"p99_ms\":" + std::to_string(stage.p99_ms);
    json += ",\"p999_ms\":" + std::to_string(stage.p999_ms) + "}";
}

struct VariantResult {
    std::size_t stream_count;
    vision_engine::VideoPipelineStats stats;
    double max_abs_error;
};

VariantResult run_variant(
    std::size_t stream_count,
    const Options& options,
    std::size_t frame_elements,
    const std::vector<std::uint8_t>& rgb_frames,
    std::vector<float>& output_frames,
    std::span<const float> reference
) {
    vision_engine::CudaVideoPipeline pipeline(
        options.width, options.height, stream_count
    );
    const std::size_t warmup_elements = options.warmup_frames * frame_elements;
    static_cast<void>(pipeline.process(
        std::span<const std::uint8_t>(rgb_frames).first(
            options.warmup_frames * frame_elements
        ),
        options.warmup_frames,
        std::span<float>(output_frames).first(warmup_elements)
    ));
    const vision_engine::VideoPipelineStats stats =
        pipeline.process(
            std::span<const std::uint8_t>(rgb_frames).first(
                options.frames * frame_elements
            ),
            options.frames,
            std::span<float>(output_frames).first(options.frames * frame_elements)
        );
    const double error = max_abs_error(
        std::span<const float>(output_frames).first(frame_elements), reference
    );
    if (error > 1e-6) {
        throw std::runtime_error("video pipeline differs from scalar reference");
    }
    return {stream_count, stats, error};
}

std::string serialize(
    const Options& options,
    const std::string& device_name,
    const std::vector<VariantResult>& results
) {
    std::string json = "{\n  \"metadata\": {\n";
    json += "    \"os\": \"" + operating_system_name() + "\",\n";
    json += "    \"architecture\": \"" + architecture_name() + "\",\n";
    json += "    \"compiler\": \"" + std::string(__VERSION__) + "\",\n";
    json += "    \"logical_cpu_count\": " +
        std::to_string(std::thread::hardware_concurrency()) + ",\n";
    json += "    \"cuda_device\": \"" + device_name + "\",\n";
    json += "    \"width\": " + std::to_string(options.width) + ",\n";
    json += "    \"height\": " + std::to_string(options.height) + ",\n";
    json += "    \"frame_count\": " + std::to_string(options.frames) + ",\n";
    json += "    \"warmup_frames\": " + std::to_string(options.warmup_frames) + ",\n";
    json += "    \"input_format\": \"contiguous RGB uint8 frames\",\n";
    json += "    \"output_format\": \"contiguous normalized float NCHW frames\",\n";
    json += "    \"cuda_timing\": \"CUDA events per stage; FPS includes CPU staging and output copies\"\n";
    json += "  },\n  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const VariantResult& result = results[index];
        json += "    {\"stream_count\":" + std::to_string(result.stream_count);
        json += ",\"cuda_active\":" +
            std::string(result.stats.cuda_active ? "true" : "false");
        json += ",\"fps\":" + std::to_string(result.stats.fps);
        const double fps_gain = result.stats.fps / results.front().stats.fps;
        json += ",\"fps_gain_vs_single_stream\":" + std::to_string(fps_gain);
        json += ",\"max_abs_error\":" + std::to_string(result.max_abs_error);
        json += ",\"stages_ms\":{";
        json += "\"host_input_copy\":";
        append_stage(json, result.stats.host_input_copy);
        json += ",\"h2d\":";
        append_stage(json, result.stats.host_to_device);
        json += ",\"normalize_kernel\":";
        append_stage(json, result.stats.normalize_kernel);
        json += ",\"d2h\":";
        append_stage(json, result.stats.device_to_host);
        json += ",\"host_output_copy\":";
        append_stage(json, result.stats.host_output_copy);
        json += ",\"end_to_end\":";
        append_stage(json, result.stats.end_to_end);
        json += ",\"cpu_fallback\":";
        append_stage(json, result.stats.cpu_fallback);
        json += "}}";
        json += index + 1 == results.size() ? "\n" : ",\n";
    }
    json += "  ]\n}\n";
    return json;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        if (options.width > std::numeric_limits<std::size_t>::max() / options.height) {
            throw std::overflow_error("frame dimensions overflow");
        }
        const std::size_t pixels = options.width * options.height;
        if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::overflow_error("frame element count overflows");
        }
        const std::size_t frame_elements = pixels * 3;
        if (frame_elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::overflow_error("frame output size overflows");
        }
        if (options.frames > std::numeric_limits<std::size_t>::max() / frame_elements ||
            options.frames > std::numeric_limits<std::size_t>::max() /
                                 frame_elements / sizeof(float)) {
            throw std::overflow_error("video buffer size overflows");
        }
        const std::size_t allocated_frames = std::max(options.frames, options.warmup_frames);
        if (allocated_frames > std::numeric_limits<std::size_t>::max() / frame_elements ||
            allocated_frames > std::numeric_limits<std::size_t>::max() /
                                   frame_elements / sizeof(float)) {
            throw std::overflow_error("video buffer size overflows");
        }
        const std::size_t input_elements = allocated_frames * frame_elements;
        const std::size_t output_elements = allocated_frames * frame_elements;
        std::vector<std::uint8_t> rgb_frames(input_elements);
        for (std::size_t index = 0; index < rgb_frames.size(); ++index) {
            rgb_frames[index] = static_cast<std::uint8_t>((index * 37 + 11) % 256);
        }
        std::vector<float> output_frames(output_elements);
        std::vector<float> reference(frame_elements);
        vision_engine::normalize_rgb_to_nchw_scalar(
            std::span<const std::uint8_t>(rgb_frames).first(frame_elements),
            options.width,
            options.height,
            reference
        );

        const VariantResult single_stream = run_variant(
            1, options, frame_elements, rgb_frames, output_frames, reference
        );
        const VariantResult two_streams = run_variant(
            2, options, frame_elements, rgb_frames, output_frames, reference
        );
        std::string device_name;
        if (two_streams.stats.cuda_active) {
            device_name = vision_engine::CudaPreprocessor::cuda_device_name();
        }
        std::cout << serialize(options, device_name, {single_stream, two_streams});
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "video_pipeline_benchmark: " << error.what() << '\n';
        return 1;
    }
}
