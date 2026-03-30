#include "sao_inference/T5Encoder.h"
#include <cassert>
#include <iostream>

namespace sao {

T5Encoder::T5Encoder(const std::string& onnx_path, bool use_cuda, bool use_coreml)
    : m_model(onnx_path, use_cuda, use_coreml)
{
}

std::vector<float> T5Encoder::encode(
    const std::vector<int64_t>& input_ids,
    const std::vector<int64_t>& attention_mask)
{
    assert(input_ids.size() == attention_mask.size());

    int64_t batch = 1;
    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    std::array<int64_t, 2> shape = {batch, seq_len};

    auto ids_tensor = Ort::Value::CreateTensor<int64_t>(
        m_model.memory_info(),
        const_cast<int64_t*>(input_ids.data()),
        input_ids.size(),
        shape.data(), shape.size()
    );

    auto mask_tensor = Ort::Value::CreateTensor<int64_t>(
        m_model.memory_info(),
        const_cast<int64_t*>(attention_mask.data()),
        attention_mask.size(),
        shape.data(), shape.size()
    );

    std::vector<const char*> input_names = {"input_ids", "attention_mask"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(ids_tensor));
    inputs.push_back(std::move(mask_tensor));

    auto outputs = m_model.run(input_names, inputs);
    assert(!outputs.empty());

    auto& out = outputs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();
    size_t total = 1;
    for (auto d : out_shape) total *= d;

    const float* data = out.GetTensorData<float>();
    return std::vector<float>(data, data + total);
}

} // namespace sao
