cmake_minimum_required(VERSION 3.25)
project(vampnet_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

option(VAMPNET_FETCH_ONNXRUNTIME_GPU
    "On Linux, FetchContent downloads onnxruntime-linux-x64-gpu (CUDA) instead of CPU-only."
    OFF)

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
        if(VAMPNET_FETCH_ONNXRUNTIME_GPU)
            set(ONNXRUNTIME_PLATFORM "linux-x64-gpu")
        else()
            set(ONNXRUNTIME_PLATFORM "linux-x64")
        endif()
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

message(STATUS "vampnet_inference: ONNX include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "vampnet_inference: ONNX library: ${ONNXRUNTIME_LIBRARY}")

add_library(vampnet_inference STATIC
    src/VampNetInference.cpp
)

target_include_directories(vampnet_inference
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(vampnet_inference
    PUBLIC
        ${ONNXRUNTIME_LIBRARY}
)

option(VAMPNET_BUILD_TESTS "Build vampnet_inference parity tests" OFF)
if(VAMPNET_BUILD_TESTS)
    enable_testing()
    message(STATUS "vampnet_inference: VAMPNET_BUILD_TESTS=ON (add tests when parity data exists)")
endif()
