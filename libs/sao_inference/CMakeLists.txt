cmake_minimum_required(VERSION 3.22)
project(sao_inference LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# --- Dependencies ---

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        ${ONNXRUNTIME_ROOT}/include
        /usr/local/include/onnxruntime
        /opt/homebrew/include/onnxruntime
    PATH_SUFFIXES onnxruntime
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        ${ONNXRUNTIME_ROOT}/lib
        /usr/local/lib
        /opt/homebrew/lib
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(STATUS "ONNX Runtime not found locally, attempting FetchContent...")

    set(ONNXRUNTIME_VERSION "1.20.1")
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
            set(ONNXRUNTIME_PLATFORM "osx-arm64")
        else()
            set(ONNXRUNTIME_PLATFORM "osx-x86_64")
        endif()
    elseif(UNIX)
        set(ONNXRUNTIME_PLATFORM "linux-x64")
    else()
        message(FATAL_ERROR "Unsupported platform for ONNX Runtime auto-download")
    endif()

    FetchContent_Declare(
        onnxruntime_prebuilt
        URL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-${ONNXRUNTIME_PLATFORM}-${ONNXRUNTIME_VERSION}.tgz"
    )
    FetchContent_MakeAvailable(onnxruntime_prebuilt)

    set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_prebuilt_SOURCE_DIR}/include")
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${onnxruntime_prebuilt_SOURCE_DIR}/lib"
        NO_DEFAULT_PATH
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# cnpy for loading .npy files
FetchContent_Declare(
    cnpy
    GIT_REPOSITORY https://github.com/rogersce/cnpy.git
    GIT_TAG master
)
FetchContent_MakeAvailable(cnpy)

# SentencePiece for T5 tokenization
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "" FORCE)
set(CMAKE_INSTALL_INCDIR "${CMAKE_BINARY_DIR}/_spm_inc" CACHE STRING "" FORCE)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.2.0
)
FetchContent_MakeAvailable(sentencepiece)

find_package(ZLIB REQUIRED)

# nlohmann/json for manifest parsing (header-only)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# --- Library target ---

add_library(sao_inference STATIC
    src/OnnxModel.cpp
    src/T5Encoder.cpp
    src/DiTModel.cpp
    src/DiTInpaintModel.cpp
    src/VAEDecoder.cpp
    src/VAEEncoder.cpp
    src/NumberEmbedder.cpp
    src/ConditioningAssembler.cpp
    src/InpaintConditioningAssembler.cpp
    src/RectifiedFlowSampler.cpp
    src/InpaintSampler.cpp
    src/Pipeline.cpp
    src/ZenonPipeline.cpp
    src/ZenonPipelineConfig.cpp
    src/Tokenizer.cpp
)

target_include_directories(sao_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    PRIVATE
        ${cnpy_SOURCE_DIR}
        ${sentencepiece_SOURCE_DIR}/src
)

target_link_libraries(sao_inference
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
        cnpy-static
        sentencepiece-static
        nlohmann_json::nlohmann_json
        ZLIB::ZLIB
)

if(APPLE)
    target_link_libraries(sao_inference PUBLIC "-framework CoreML" "-framework Foundation")
endif()

# --- CLI app (when built standalone) ---

option(SAO_BUILD_CLI "Build the sao_cli app" ON)

if(SAO_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../apps/sao_cli/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../apps/sao_cli ${CMAKE_BINARY_DIR}/sao_cli)
endif()

option(ZENON_BUILD_CLI "Build the zenon_cli app" ON)

if(ZENON_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../apps/zenon_cli/CMakeLists.txt")
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../apps/zenon_cli ${CMAKE_BINARY_DIR}/zenon_cli)
endif()

# --- Parity tests ---

option(SAO_BUILD_TESTS "Build parity tests" OFF)

if(SAO_BUILD_TESTS)
    enable_testing()

    set(SAO_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

    function(add_parity_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE sao_inference cnpy-static)
        target_include_directories(${TEST_NAME} PRIVATE ${cnpy_SOURCE_DIR})
        target_compile_definitions(${TEST_NAME} PRIVATE
            PARITY_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/parity"
            ONNX_MODEL_DIR="${SAO_REPO_ROOT}/models/onnx"
            WEIGHTS_DIR="${SAO_REPO_ROOT}/models/weights"
        )
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_parity_test(test_number_embedder_parity tests/test_number_embedder_parity.cpp)
    add_parity_test(test_t5_parity tests/test_t5_parity.cpp)
    add_parity_test(test_conditioning_parity tests/test_conditioning_parity.cpp)
    add_parity_test(test_vae_parity tests/test_vae_parity.cpp)
    add_parity_test(test_dit_parity tests/test_dit_parity.cpp)
    add_parity_test(test_sampler_parity tests/test_sampler_parity.cpp)
    add_parity_test(test_full_pipeline_parity tests/test_full_pipeline_parity.cpp)

    # --- Zenon parity tests ---
    function(add_zenon_parity_test TEST_NAME TEST_SOURCE)
        add_executable(${TEST_NAME} ${TEST_SOURCE})
        target_link_libraries(${TEST_NAME} PRIVATE sao_inference cnpy-static)
        target_include_directories(${TEST_NAME} PRIVATE ${cnpy_SOURCE_DIR})
        target_compile_definitions(${TEST_NAME} PRIVATE
            ZENON_PARITY_DATA_DIR="${SAO_REPO_ROOT}/tests/parity_data_zenon"
            ZENON_ONNX_DIR="${SAO_REPO_ROOT}/models/zenon/onnx"
            ZENON_WEIGHTS_DIR="${SAO_REPO_ROOT}/models/zenon/weights"
            ZENON_MANIFEST_PATH="${SAO_REPO_ROOT}/models/zenon/zenon_pipeline_manifest.json"
        )
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endfunction()

    add_zenon_parity_test(test_zenon_vae_encoder_parity tests/test_zenon_vae_encoder_parity.cpp)
    add_zenon_parity_test(test_zenon_dit_parity tests/test_zenon_dit_parity.cpp)
    add_zenon_parity_test(test_zenon_conditioning_parity tests/test_zenon_conditioning_parity.cpp)
    add_zenon_parity_test(test_zenon_sampler_parity tests/test_zenon_sampler_parity.cpp)
    add_zenon_parity_test(test_zenon_full_pipeline_parity tests/test_zenon_full_pipeline_parity.cpp)
endif()
