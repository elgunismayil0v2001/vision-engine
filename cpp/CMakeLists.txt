cmake_minimum_required(VERSION 3.20)
project(vision_engine LANGUAGES CXX)

option(ENABLE_SANITIZERS "Build C++ targets with AddressSanitizer and UBSan" OFF)
option(ENABLE_OPENMP "Build the OpenMP preprocessing path when available" ON)
option(ENABLE_CUDA "Build the optional CUDA preprocessing path" OFF)
set(ONNXRUNTIME_VERSION "1.31.1" CACHE STRING "Required ONNX Runtime SDK version")
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of the ONNX Runtime ${ONNXRUNTIME_VERSION} C++ SDK")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include"
    REQUIRED
)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib"
    REQUIRED
)

if(ENABLE_SANITIZERS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address,undefined)
    else()
        message(FATAL_ERROR "ENABLE_SANITIZERS requires Clang or GCC")
    endif()
endif()

add_library(ONNXRuntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(ONNXRuntime::onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

add_library(vision_engine
    src/onnx_classifier.cpp
    src/preprocess.cpp
    src/video_pipeline_fallback.cpp
)
target_include_directories(vision_engine PUBLIC include)
target_compile_features(vision_engine PUBLIC cxx_std_20)
target_compile_options(vision_engine PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
target_compile_definitions(vision_engine PRIVATE
    VISION_ENGINE_ONNXRUNTIME_VERSION="${ONNXRUNTIME_VERSION}"
)
target_link_libraries(vision_engine PUBLIC ONNXRuntime::onnxruntime)

if(ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(vision_engine PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(vision_engine PUBLIC VISION_ENGINE_HAS_OPENMP=1)
    endif()
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_sources(vision_engine PRIVATE
        src/preprocess_cuda.cu
        src/video_pipeline_cuda.cu
    )
    target_compile_definitions(vision_engine PUBLIC VISION_ENGINE_HAS_CUDA=1)
    target_link_libraries(vision_engine PUBLIC CUDA::cudart)
    set_target_properties(vision_engine PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_executable(video_pipeline_benchmark src/video_pipeline_benchmark.cpp)
    target_compile_features(video_pipeline_benchmark PRIVATE cxx_std_20)
    target_compile_options(video_pipeline_benchmark PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(video_pipeline_benchmark PRIVATE vision_engine)
    if(UNIX)
        set_target_properties(video_pipeline_benchmark PROPERTIES
            BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
        )
    endif()
endif()

add_executable(vision_engine_cli src/main.cpp)
target_compile_features(vision_engine_cli PRIVATE cxx_std_20)
target_compile_options(vision_engine_cli PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(vision_engine_cli PRIVATE vision_engine)

add_executable(vision_engine_allocation_probe tests/allocation_probe.cpp)
target_compile_features(vision_engine_allocation_probe PRIVATE cxx_std_20)
target_compile_options(vision_engine_allocation_probe PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(vision_engine_allocation_probe PRIVATE vision_engine)

if(UNIX)
    set_target_properties(vision_engine_cli PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
endif()

add_executable(preprocess_benchmark src/preprocess_benchmark.cpp)
target_compile_features(preprocess_benchmark PRIVATE cxx_std_20)
target_compile_options(preprocess_benchmark PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(preprocess_benchmark PRIVATE vision_engine)
if(UNIX)
    set_target_properties(preprocess_benchmark PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    )
endif()

include(CTest)
if(BUILD_TESTING)
    find_package(GTest 1.15.2 EXACT CONFIG QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.15.2
            GIT_SHALLOW TRUE
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(vision_engine_tests tests/preprocess_test.cpp)
    target_compile_features(vision_engine_tests PRIVATE cxx_std_20)
    target_compile_options(vision_engine_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(vision_engine_tests PRIVATE vision_engine GTest::gtest_main)
    add_test(NAME vision_engine_tests COMMAND vision_engine_tests)
    add_executable(video_pipeline_tests tests/video_pipeline_test.cpp)
    target_compile_features(video_pipeline_tests PRIVATE cxx_std_20)
    target_compile_options(video_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(video_pipeline_tests PRIVATE vision_engine GTest::gtest_main)
    add_test(NAME video_pipeline_tests COMMAND video_pipeline_tests)
    add_test(
        NAME preprocess_benchmark_smoke
        COMMAND preprocess_benchmark
            --width 19 --height 7 --warmup 1 --iterations 3 --threads 2
    )

    if(ENABLE_CUDA)
        add_executable(vision_engine_cuda_tests tests/preprocess_cuda_test.cpp)
        target_compile_features(vision_engine_cuda_tests PRIVATE cxx_std_20)
        target_compile_options(vision_engine_cuda_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_link_libraries(vision_engine_cuda_tests PRIVATE vision_engine GTest::gtest_main)
        add_test(NAME vision_engine_cuda_tests COMMAND vision_engine_cuda_tests)

        add_test(
            NAME video_pipeline_benchmark_smoke
            COMMAND video_pipeline_benchmark
                --width 17 --height 13 --frames 4 --warmup-frames 1
        )
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(
        NAME onnx_cpp_parity
        COMMAND "${Python3_EXECUTABLE}"
            "${PROJECT_SOURCE_DIR}/../tests/test_cpp_parity.py"
            --binary "$<TARGET_FILE:vision_engine_cli>"
            --allocation-binary "$<TARGET_FILE:vision_engine_allocation_probe>"
    )
    set_tests_properties(onnx_cpp_parity PROPERTIES
        ENVIRONMENT "PYTHONPATH=${PROJECT_SOURCE_DIR}/.."
    )
endif()
