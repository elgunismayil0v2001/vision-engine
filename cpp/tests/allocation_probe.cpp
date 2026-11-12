#include "vision_engine/onnx_classifier.hpp"

#include <atomic>
#include <charconv>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

std::atomic<bool> g_count_allocations{false};
std::atomic<std::size_t> g_allocation_count{0};

void count_allocation() noexcept {
    if (g_count_allocations.load(std::memory_order_relaxed)) {
        g_allocation_count.fetch_add(1, std::memory_order_relaxed);
    }
}

class AllocationCounterScope {
public:
    AllocationCounterScope() {
        g_allocation_count.store(0, std::memory_order_relaxed);
        g_count_allocations.store(true, std::memory_order_relaxed);
    }
    ~AllocationCounterScope() {
        g_count_allocations.store(false, std::memory_order_relaxed);
    }

    AllocationCounterScope(const AllocationCounterScope&) = delete;
    AllocationCounterScope& operator=(const AllocationCounterScope&) = delete;
};

std::size_t parse_size(std::string_view text) {
    std::size_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size() || value == 0) {
        throw std::invalid_argument("image size and class count must be positive integers");
    }
    return value;
}

}  // namespace

void* operator new(std::size_t size) {
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        count_allocation();
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete[](void* memory) noexcept {
    ::operator delete(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    ::operator delete(memory);
}

void operator delete[](void* memory, std::size_t) noexcept {
    ::operator delete(memory);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
    void* memory = nullptr;
    const auto aligned_size = size == 0 ? static_cast<std::size_t>(alignment) : size;
    if (posix_memalign(&memory, static_cast<std::size_t>(alignment), aligned_size) != 0) {
        throw std::bad_alloc();
    }
    count_allocation();
    return memory;
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
    return ::operator new(size, alignment);
}

void operator delete(void* memory, std::align_val_t) noexcept {
    ::operator delete(memory);
}

void operator delete[](void* memory, std::align_val_t) noexcept {
    ::operator delete(memory);
}

void operator delete(void* memory, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(memory, alignment);
}

void operator delete[](void* memory, std::size_t, std::align_val_t alignment) noexcept {
    ::operator delete(memory, alignment);
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: vision_engine_allocation_probe <model.onnx> <image_size> "
                         "<classes> <max_batch_size>\n";
            return 2;
        }
        const auto image_size = parse_size(argv[2]);
        const auto num_classes = parse_size(argv[3]);
        const auto max_batch_size = parse_size(argv[4]);
        if (image_size > std::numeric_limits<std::size_t>::max() / image_size) {
            throw std::overflow_error("image tensor size overflows");
        }
        const auto pixels = image_size * image_size;
        if (pixels > std::numeric_limits<std::size_t>::max() / 3) {
            throw std::overflow_error("image tensor size overflows");
        }
        const auto input_size = 3 * pixels;
        if (max_batch_size > std::numeric_limits<std::size_t>::max() / input_size ||
            max_batch_size > std::numeric_limits<std::size_t>::max() / num_classes) {
            throw std::overflow_error("batch tensor size overflows");
        }
        std::vector<float> input(input_size * max_batch_size, 0.125F);
        std::vector<float> output(num_classes * max_batch_size);
        vision_engine::OnnxClassifier classifier(
            argv[1], image_size, num_classes, max_batch_size, 1
        );

        constexpr std::size_t kWarmupIterations = 20;
        constexpr std::size_t kTimedIterations = 100;
        for (std::size_t batch = 1; batch <= max_batch_size; ++batch) {
            const auto input_count = batch * input_size;
            const auto output_count = batch * num_classes;
            const std::span<const float> batch_input(input.data(), input_count);
            const std::span<float> batch_output(output.data(), output_count);
            for (std::size_t iteration = 0; iteration < kWarmupIterations; ++iteration) {
                classifier.run(batch_input, batch, batch_output);
            }
        }
        {
            const AllocationCounterScope counter_scope;
            for (std::size_t iteration = 0; iteration < kTimedIterations; ++iteration) {
                const auto batch = iteration % max_batch_size + 1;
                const auto input_count = batch * input_size;
                const auto output_count = batch * num_classes;
                const std::span<const float> batch_input(input.data(), input_count);
                const std::span<float> batch_output(output.data(), output_count);
                classifier.run(batch_input, batch, batch_output);
            }
        }
        const auto allocations = g_allocation_count.load(std::memory_order_relaxed);
        std::cout << allocations << " C++ operator new allocations across "
                  << kTimedIterations << " warmed inference calls\n";
        return 0;
    } catch (const std::exception& error) {
        g_count_allocations.store(false, std::memory_order_relaxed);
        std::cerr << "vision_engine_allocation_probe: " << error.what() << '\n';
        return 1;
    }
}
