#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>

namespace vision_engine {

class OnnxClassifier {
public:
    OnnxClassifier(
        const std::filesystem::path& model_path,
        std::size_t image_size,
        std::size_t num_classes,
        std::size_t max_batch_size = 1,
        int intra_op_threads = 1
    );
    ~OnnxClassifier();

    OnnxClassifier(OnnxClassifier&&) noexcept;
    OnnxClassifier& operator=(OnnxClassifier&&) noexcept;
    OnnxClassifier(const OnnxClassifier&) = delete;
    OnnxClassifier& operator=(const OnnxClassifier&) = delete;

    // Uses caller-provided spans and reusable tensors allocated for batches [1, max_batch_size].
    // Calls on one instance must be serialized because its preallocated buffers are reused.
    void run(
        std::span<const float> nchw,
        std::size_t batch,
        std::span<float> logits
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vision_engine
