#include "vampnet_inference/VampNetInference.hpp"

namespace vampnet {

namespace {

const char kVersion[] = "vampnet_inference 0.0.1 (stub)";

}  // namespace

const char* version_string() {
    return kVersion;
}

struct VampNetInference::Impl {
    // ONNX sessions and paths will live here.
};

VampNetInference::VampNetInference() : impl_(new Impl) {}

VampNetInference::~VampNetInference() {
    delete impl_;
    impl_ = nullptr;
}

bool VampNetInference::is_ready() const {
    return false;
}

}  // namespace vampnet
