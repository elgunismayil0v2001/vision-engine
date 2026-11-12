#include "vision_engine/preprocess.hpp"

#ifdef VISION_ENGINE_HAS_CUDA
#include "vision_engine/preprocess_cuda.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef VISION_ENGINE_HAS_OPENMP
#include <omp.h>
#endif

namespace {

struct BenchmarkOptions {
    std::size_t width = 640;
    std::size_t height = 480;
    std::size_t warmup = 20;
    std::size_t iterations = 200;
    int threads = 4;
    std::string output_path;
};

struct Latencies {
    double p50_ms;
    double p99_ms;
    double p999_ms;
};

struct Result {
    std::string name;
    int threads_used;
    Latencies latency;
    double max_abs_error;
    double speedup_vs_scalar;
};

std::size_t parse_size(const std::string& text, const char* name, bool allow_zero = false) {
    if (text.empty() || text.front() == '-') {
        throw std::invalid_argument(std::string(name) + " must be a valid integer");
    }
    std::size_t parsed_characters = 0;
    const unsigned long long parsed = std::stoull(text, &parsed_characters);
    if (parsed_characters != text.size() || (!allow_zero && parsed == 0) ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(
            std::string(name) + (allow_zero ? " must be nonnegative" : " must be positive")
        );
    }
    return static_cast<std::size_t>(parsed);
}

BenchmarkOptions parse_args(int argc, char** argv) {
    BenchmarkOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + argument);
        }
        const std::string value = argv[++index];
        if (argument == "--width") {
            options.width = parse_size(value, "width");
        } else if (argument == "--height") {
            options.height = parse_size(value, "height");
        } else if (argument == "--warmup") {
            options.warmup = parse_size(value, "warmup", true);
        } else if (argument == "--iterations") {
            options.iterations = parse_size(value, "iterations");
        } else if (argument == "--threads") {
            const std::size_t threads = parse_size(value, "threads");
            if (threads > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("threads exceeds the supported integer range");
            }
            options.threads = static_cast<int>(threads);
        } else if (argument == "--output") {
            options.output_path = value;
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.width > std::numeric_limits<std::size_t>::max() / options.height ||
        options.width * options.height > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error("image dimensions overflow");
    }
    return options;
}

double percentile(std::vector<double> values, double percentile_value) {
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(percentile_value * static_cast<double>(values.size()));
    const std::size_t index = static_cast<std::size_t>(rank) - 1;
    return values[index];
}

template <typename Function>
Latencies measure(
    Function&& function,
    std::size_t warmup_iterations,
    std::size_t timed_iterations
) {
    for (std::size_t iteration = 0; iteration < warmup_iterations; ++iteration) {
        function();
    }
    std::vector<double> durations;
    durations.reserve(timed_iterations);
    for (std::size_t iteration = 0; iteration < timed_iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        function();
        const auto finish = std::chrono::steady_clock::now();
        const std::chrono::duration<double, std::milli> elapsed = finish - start;
        durations.push_back(elapsed.count());
    }
    return {
        percentile(durations, 0.50),
        percentile(durations, 0.99),
        percentile(durations, 0.999),
    };
}

double max_abs_error(const std::vector<float>& actual, const std::vector<float>& expected) {
    double maximum = 0.0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(actual[index]) - expected[index])
        );
    }
    return maximum;
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
#elif defined(_WIN32)
    return "Windows";
#else
    return "unknown";
#endif
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char character : value) {
        if (character == '\\' || character == '"') {
            escaped.push_back('\\');
        }
        if (character == '\n') {
            escaped += "\\n";
        } else {
            escaped.push_back(character);
        }
    }
    return escaped;
}

std::string serialize_results(
    const BenchmarkOptions& options,
    const std::vector<Result>& results,
    bool cuda_compiled,
    bool cuda_active,
    const std::string& cuda_name
) {
    std::string json = "{\n  \"metadata\": {\n";
    json += "    \"os\": \"" + operating_system_name() + "\",\n";
    json += "    \"architecture\": \"" + architecture_name() + "\",\n";
    json += "    \"compiler\": \"" + json_escape(__VERSION__) + "\",\n";
    json += "    \"logical_cpu_count\": " +
        std::to_string(std::thread::hardware_concurrency()) + ",\n";
    json += "    \"omp_enabled\": " +
        std::string(vision_engine::openmp_available() ? "true" : "false") + ",\n";
    json += "    \"neon_enabled\": " +
        std::string(vision_engine::neon_available() ? "true" : "false") + ",\n";
    json += "    \"cuda_compiled\": " +
        std::string(cuda_compiled ? "true" : "false") + ",\n";
    json += "    \"cuda_active\": " +
        std::string(cuda_active ? "true" : "false") + ",\n";
    json += "    \"cuda_device\": \"" + json_escape(cuda_name) + "\",\n";
    json += "    \"width\": " + std::to_string(options.width) + ",\n";
    json += "    \"height\": " + std::to_string(options.height) + ",\n";
    json += "    \"thread_count\": " + std::to_string(options.threads) + ",\n";
    json += "    \"warmup_iterations\": " + std::to_string(options.warmup) + ",\n";
    json += "    \"timed_iterations\": " + std::to_string(options.iterations) + ",\n";
    json += "    \"input_pattern\": \"rgb[index] = (index * 37 + 11) % 256\",\n";
    json += "    \"openmp_schedule\": \"static\",\n";
    json += "    \"timing_method\": \"steady_clock per call; CUDA includes H2D/D2H copies\"\n";
    json += "  },\n  \"results\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Result& result = results[index];
        json += "    {\"kernel\": \"" + result.name + "\", ";
        json += "\"threads_used\": " + std::to_string(result.threads_used) + ", ";
        json += "\"p50_ms\": " + std::to_string(result.latency.p50_ms) + ", ";
        json += "\"p99_ms\": " + std::to_string(result.latency.p99_ms) + ", ";
        json += "\"p999_ms\": " + std::to_string(result.latency.p999_ms) + ", ";
        json += "\"max_abs_error\": " + std::to_string(result.max_abs_error) + ", ";
        json += "\"speedup_vs_scalar\": " +
            std::to_string(result.speedup_vs_scalar) + "}";
        json += index + 1 == results.size() ? "\n" : ",\n";
    }
    json += "  ]\n}\n";
    return json;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const BenchmarkOptions options = parse_args(argc, argv);
#ifdef VISION_ENGINE_HAS_OPENMP
        omp_set_dynamic(0);
#endif
        const std::size_t pixels = options.width * options.height;
        const std::size_t element_count = pixels * 3;
        std::vector<std::uint8_t> input(element_count);
        for (std::size_t index = 0; index < input.size(); ++index) {
            input[index] = static_cast<std::uint8_t>((index * 37 + 11) % 256);
        }

        std::vector<float> reference(element_count);
        std::vector<float> output(element_count);
        vision_engine::normalize_rgb_to_nchw_scalar(
            input, options.width, options.height, reference
        );
        std::vector<Result> results;
        const auto add_result = [&](const std::string& name,
                                    int threads_used,
                                    auto&& function) {
            function();
            const double error = max_abs_error(output, reference);
            if (error > 1e-6) {
                throw std::runtime_error(name + " differs from scalar reference");
            }
            const Latencies latency =
                measure(function, options.warmup, options.iterations);
            const double scalar_latency = results.empty() ? latency.p50_ms
                                                          : results.front().latency.p50_ms;
            results.push_back(
                {name, threads_used, latency, error, scalar_latency / latency.p50_ms}
            );
        };

        add_result("scalar", 1, [&] {
            vision_engine::normalize_rgb_to_nchw_scalar(
                input, options.width, options.height, output
            );
        });
        add_result(
            vision_engine::openmp_available() ? "openmp_static" : "openmp_scalar_fallback",
            vision_engine::openmp_available() ? options.threads : 1,
            [&] {
                vision_engine::normalize_rgb_to_nchw_openmp(
                    input, options.width, options.height, output, options.threads
                );
            }
        );
        add_result(
            vision_engine::neon_available() ? "neon" : "neon_scalar_fallback",
            1,
            [&] {
            vision_engine::normalize_rgb_to_nchw_neon(
                input, options.width, options.height, output
            );
            }
        );

        bool cuda_compiled = false;
        bool cuda_active = false;
        std::string cuda_name;
#ifdef VISION_ENGINE_HAS_CUDA
        cuda_compiled = true;
        vision_engine::CudaPreprocessor cuda_processor(options.width, options.height);
        cuda_active = cuda_processor.using_cuda();
        if (cuda_active) {
            cuda_name = vision_engine::CudaPreprocessor::cuda_device_name();
        }
        add_result(
            cuda_active ? "cuda_with_transfers" : "cuda_scalar_fallback",
            1,
            [&] { cuda_processor.process(input, output); }
        );
#endif

        const std::string json = serialize_results(
            options, results, cuda_compiled, cuda_active, cuda_name
        );
        if (options.output_path.empty()) {
            std::cout << json;
        } else {
            std::ofstream output_file(options.output_path);
            if (!output_file) {
                throw std::runtime_error("could not open benchmark output file");
            }
            output_file << json;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "preprocess_benchmark: " << error.what() << '\n';
        return 1;
    }
}
