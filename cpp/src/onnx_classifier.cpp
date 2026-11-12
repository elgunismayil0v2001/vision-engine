#include "vision_engine/onnx_classifier.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace vision_engine {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b, const char* label) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw std::overflow_error(std::string(label) + " size overflows");
    }
    return a * b;
}

std::int64_t to_dimension(std::size_t value, const char* label) {
    if (value == 0 || value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::invalid_argument(std::string(label) + " must be a positive int64 dimension");
    }
    return static_cast<std::int64_t>(value);
}

}  // namespace

struct OnnxClassifier::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "vision_engine"};
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::RunOptions run_options{nullptr};
    std::string input_name;
    std::string output_name;
    std::size_t image_size;
    std::size_t num_classes;
    std::size_t max_batch_size;
    std::vector<std::vector<float>> input_buffers;
    std::vector<std::vector<float>> output_buffers;
    std::vector<Ort::Value> input_tensors;
    std::vector<Ort::Value> output_tensors;

    Impl(
        const std::filesystem::path& model_path,
        std::size_t expected_image_size,
        std::size_t expected_num_classes,
        std::size_t expected_max_batch_size,
        int intra_op_threads
    ) : image_size(expected_image_size),
        num_classes(expected_num_classes),
        max_batch_size(expected_max_batch_size) {
        const std::string runtime_version = OrtGetApiBase()->GetVersionString();
        if (runtime_version != VISION_ENGINE_ONNXRUNTIME_VERSION) {
            throw std::runtime_error(
                "ONNX Runtime SDK version mismatch: expected " +
                std::string(VISION_ENGINE_ONNXRUNTIME_VERSION) + ", got " + runtime_version
            );
        }
        if (intra_op_threads < 1) {
            throw std::invalid_argument("intra-op thread count must be positive");
        }
        (void)to_dimension(image_size, "image size");
        (void)to_dimension(num_classes, "class count");
        (void)to_dimension(max_batch_size, "maximum batch size");

        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(intra_op_threads);
        options.SetInterOpNumThreads(1);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session = Ort::Session(env, model_path.c_str(), options);

        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::invalid_argument("classifier model must have one input and one output");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        auto allocated_input_name = session.GetInputNameAllocated(0, allocator);
        auto allocated_output_name = session.GetOutputNameAllocated(0, allocator);
        input_name = allocated_input_name.get();
        output_name = allocated_output_name.get();
        if (input_name != "input" || output_name != "logits") {
            throw std::invalid_argument("model input/output names must be 'input' and 'logits'");
        }

        const auto input_type_info = session.GetInputTypeInfo(0);
        const auto output_type_info = session.GetOutputTypeInfo(0);
        const auto input_info = input_type_info.GetTensorTypeAndShapeInfo();
        const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
        const auto input_type = input_info.GetElementType();
        const auto output_type = output_info.GetElementType();
        if (input_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            throw std::invalid_argument(
                "classifier input and output tensors must be float32 (got " +
                std::to_string(static_cast<int>(input_type)) + " and " +
                std::to_string(static_cast<int>(output_type)) + ")"
            );
        }
        const auto input_shape = input_info.GetShape();
        const auto output_shape = output_info.GetShape();
        const auto image_dimension = to_dimension(image_size, "image size");
        const auto class_dimension = to_dimension(num_classes, "class count");
        if (input_shape.size() != 4 || input_shape[0] != -1 || input_shape[1] != 3 ||
            input_shape[2] != image_dimension || input_shape[3] != image_dimension) {
            throw std::invalid_argument("model input must be dynamic-batch NCHW with 3 image channels");
        }
        if (output_shape.size() != 2 || output_shape[0] != -1 || output_shape[1] != class_dimension) {
            throw std::invalid_argument("model output must be dynamic-batch [N, num_classes] logits");
        }

        const std::size_t input_per_sample = checked_product(
            checked_product(3, image_size, "input"), image_size, "input"
        );
        input_buffers.reserve(max_batch_size);
        output_buffers.reserve(max_batch_size);
        input_tensors.reserve(max_batch_size);
        output_tensors.reserve(max_batch_size);
        for (std::size_t batch = 1; batch <= max_batch_size; ++batch) {
            const auto input_count = checked_product(batch, input_per_sample, "input");
            const auto output_count = checked_product(batch, num_classes, "output");
            input_buffers.emplace_back(input_count, 0.0F);
            output_buffers.emplace_back(output_count, 0.0F);

            const std::array<std::int64_t, 4> input_dimensions{
                to_dimension(batch, "batch size"), 3, image_dimension, image_dimension
            };
            const std::array<std::int64_t, 2> output_dimensions{
                to_dimension(batch, "batch size"), class_dimension
            };
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                input_buffers.back().data(),
                input_buffers.back().size(),
                input_dimensions.data(),
                input_dimensions.size()
            ));
            output_tensors.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                output_buffers.back().data(),
                output_buffers.back().size(),
                output_dimensions.data(),
                output_dimensions.size()
            ));
        }
    }

    void run(std::span<const float> nchw, std::size_t batch, std::span<float> logits) {
        if (batch == 0 || batch > max_batch_size) {
            throw std::invalid_argument("batch size must be in [1, max_batch_size]");
        }
        const std::size_t input_per_sample = checked_product(
            checked_product(3, image_size, "input"), image_size, "input"
        );
        const std::size_t expected_input = checked_product(batch, input_per_sample, "input");
        const std::size_t expected_output = checked_product(batch, num_classes, "output");
        if (nchw.size() != expected_input || logits.size() != expected_output) {
            throw std::invalid_argument("input or output buffer size does not match the model shape");
        }

        const auto buffer_index = batch - 1;
        std::copy(nchw.begin(), nchw.end(), input_buffers[buffer_index].begin());
        const char* input_names[] = {input_name.c_str()};
        const char* output_names[] = {output_name.c_str()};
        session.Run(
            run_options,
            input_names,
            &input_tensors[buffer_index],
            1,
            output_names,
            &output_tensors[buffer_index],
            1
        );
        std::copy(
            output_buffers[buffer_index].begin(), output_buffers[buffer_index].end(), logits.begin()
        );
    }
};

OnnxClassifier::OnnxClassifier(
    const std::filesystem::path& model_path,
    std::size_t image_size,
    std::size_t num_classes,
    std::size_t max_batch_size,
    int intra_op_threads
) : impl_(std::make_unique<Impl>(
        model_path, image_size, num_classes, max_batch_size, intra_op_threads
    )) {}

OnnxClassifier::~OnnxClassifier() = default;
OnnxClassifier::OnnxClassifier(OnnxClassifier&&) noexcept = default;
OnnxClassifier& OnnxClassifier::operator=(OnnxClassifier&&) noexcept = default;

void OnnxClassifier::run(
    std::span<const float> nchw,
    std::size_t batch,
    std::span<float> logits
) {
    impl_->run(nchw, batch, logits);
}

}  // namespace vision_engine
