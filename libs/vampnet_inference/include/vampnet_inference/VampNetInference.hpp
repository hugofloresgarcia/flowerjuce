#pragma once

#include <cstddef>
#include <string>

namespace vampnet {

/**
 * Returns a short version string for the vampnet_inference library (build id).
 */
const char* version_string();

/**
 * Placeholder for the ONNX-based VampNet pipeline. Loads exported graphs from
 * `models/` (manifest TBD). Not yet wired to real weights — API will expand
 * with encoder, codec, masked transformer steps, and iterative sampling.
 */
class VampNetInference {
public:
    VampNetInference();
    ~VampNetInference();

    VampNetInference(const VampNetInference&) = delete;
    VampNetInference& operator=(const VampNetInference&) = delete;
    VampNetInference(VampNetInference&&) = delete;
    VampNetInference& operator=(VampNetInference&&) = delete;

    /**
     * True once ONNX sessions and manifest paths are validated (stub returns false until implemented).
     */
    bool is_ready() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace vampnet
