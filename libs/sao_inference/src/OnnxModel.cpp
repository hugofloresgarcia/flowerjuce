#include "sao_inference/OnnxModel.h"
#include <iostream>
#include <unordered_map>

#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

namespace sao {

OnnxModel::OnnxModel(const std::string& model_path, bool use_cuda, bool use_coreml, bool use_migraphx)
    : m_env(ORT_LOGGING_LEVEL_WARNING, "sao_inference")
    , m_memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    m_session_options.SetIntraOpNumThreads(0);
    m_session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        m_session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    if (use_migraphx) {
        auto* migraphx_status = OrtSessionOptionsAppendExecutionProvider_MIGraphX(m_session_options, 0);
        if (migraphx_status != nullptr) {
            const char* msg = OrtGetApiBase()->GetApi(ORT_API_VERSION)->GetErrorMessage(migraphx_status);
            std::cerr << "[sao::OnnxModel] MIGraphX EP failed: " << msg << std::endl;
            OrtGetApiBase()->GetApi(ORT_API_VERSION)->ReleaseStatus(migraphx_status);
            assert(false && "Failed to append MIGraphX execution provider");
        }
        std::cout << "[sao::OnnxModel] MIGraphX EP enabled (device 0)" << std::endl;
    }

#ifdef __APPLE__
    if (use_coreml) {
        uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        coreml_flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
        coreml_flags |= COREML_FLAG_CREATE_MLPROGRAM;
        auto* status = OrtSessionOptionsAppendExecutionProvider_CoreML(m_session_options, coreml_flags);
        assert(status == nullptr && "Failed to append CoreML execution provider");
        std::cout << "[sao::OnnxModel] CoreML EP enabled (MLProgram, subgraphs, all compute units)" << std::endl;
    }
#else
    if (use_coreml) {
        std::cout << "[sao::OnnxModel] CoreML EP requested but not available on this platform" << std::endl;
    }
#endif

    m_session = std::make_unique<Ort::Session>(m_env, model_path.c_str(), m_session_options);

    // Cache output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_outputs = m_session->GetOutputCount();
    m_output_names_owned.reserve(num_outputs);
    m_output_names.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = m_session->GetOutputNameAllocated(i, allocator);
        m_output_names_owned.push_back(name.get());
        m_output_names.push_back(m_output_names_owned.back().c_str());
    }

    std::cout << "[sao::OnnxModel] Loaded " << model_path
              << " (" << m_session->GetInputCount() << " inputs, "
              << num_outputs << " outputs)" << std::endl;
}

std::vector<Ort::Value> OnnxModel::run(
    const std::vector<const char*>& input_names,
    const std::vector<Ort::Value>& input_tensors)
{
    assert(input_names.size() == input_tensors.size());

    return m_session->Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_tensors.data(),
        input_tensors.size(),
        m_output_names.data(),
        m_output_names.size()
    );
}

} // namespace sao
