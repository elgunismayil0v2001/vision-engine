#include "vision_engine/onnx_classifier.hpp"
#include "vision_engine/preprocess.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::size_t parse_size(std::string_view text, const char* label) {
    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() ||
        value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + label);
    }
    return static_cast<std::size_t>(value);
}

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open input file: " + path.string());
    }
    const auto byte_count = input.tellg();
    if (byte_count < 0 || static_cast<std::uint64_t>(byte_count) % sizeof(T) != 0) {
        throw std::runtime_error("input file has an invalid byte count: " + path.string());
    }
    std::vector<T> values(static_cast<std::size_t>(byte_count) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), byte_count);
    if (!input) {
        throw std::runtime_error("failed to read input file: " + path.string());
    }
    return values;
}

template <typename T>
void write_binary(const std::filesystem::path& path, std::span<const T> values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open output file: " + path.string());
    }
    const auto byte_count = values.size_bytes();
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(byte_count));
    if (!output) {
        throw std::runtime_error("failed to write output file: " + path.string());
    }
}

void print_usage(const char* program) {
    std::cerr << "Usage:\n"
              << "  " << program << " preprocess <width> <height> <rgb.bin> <nchw.f32>\n"
              << "  " << program
              << " infer <model.onnx> <image_size> <classes> <batch> <input.f32> <logits.f32>\n";
}

int run_preprocess(int argc, char** argv) {
    if (argc != 6) {
        throw std::invalid_argument("preprocess expects four arguments");
    }
    const auto width = parse_size(argv[2], "width");
    const auto height = parse_size(argv[3], "height");
    auto rgb = read_binary<std::uint8_t>(argv[4]);
    if (width != 0 && height > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("image dimensions overflow");
    }
    const auto pixels = width * height;
    if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
        throw std::overflow_error("image element count overflows");
    }
    std::vector<float> output(pixels * 3);
    vision_engine::normalize_rgb_to_nchw(rgb, width, height, output);
    write_binary<float>(argv[5], output);
    return 0;
}

int run_inference(int argc, char** argv) {
    if (argc != 8) {
        throw std::invalid_argument("infer expects six arguments");
    }
    const auto image_size = parse_size(argv[3], "image size");
    const auto num_classes = parse_size(argv[4], "class count");
    const auto batch = parse_size(argv[5], "batch size");
    auto input = read_binary<float>(argv[6]);
    if (batch != 0 && num_classes > std::numeric_limits<std::size_t>::max() / batch) {
        throw std::overflow_error("output element count overflows");
    }
    std::vector<float> output(batch * num_classes);
    vision_engine::OnnxClassifier classifier(argv[2], image_size, num_classes, batch);
    classifier.run(input, batch, output);
    write_binary<float>(argv[7], output);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            print_usage(argv[0]);
            return 2;
        }
        const std::string_view command(argv[1]);
        if (command == "preprocess") {
            return run_preprocess(argc, argv);
        }
        if (command == "infer") {
            return run_inference(argc, argv);
        }
        print_usage(argv[0]);
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "vision_engine_cli: " << error.what() << '\n';
        return 1;
    }
}
