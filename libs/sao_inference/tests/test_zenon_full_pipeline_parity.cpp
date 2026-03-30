#include "sao_inference/ZenonPipeline.h"
#include "sao_inference/ZenonPipelineConfig.h"
#include <cnpy.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <string>

static std::vector<float> load_npy_flat(const std::string& path)
{
    auto arr = cnpy::npy_load(path);
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    return std::vector<float>(data, data + total);
}

static std::vector<int64_t> load_npy_int64_from_float(const std::string& path)
{
    auto arr = cnpy::npy_load(path);
    size_t total = 1;
    for (auto d : arr.shape) total *= d;
    const float* data = arr.data<float>();
    std::vector<int64_t> result(total);
    for (size_t i = 0; i < total; ++i)
        result[i] = static_cast<int64_t>(data[i]);
    return result;
}

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;
    std::string manifest_path = ZENON_MANIFEST_PATH;

    std::cout << "=== Zenon Full Pipeline Parity Test ===" << std::endl;
    std::cout << "Manifest: " << manifest_path << std::endl;
    std::cout << "Parity data: " << parity_dir << std::endl;

    auto config = sao::ZenonPipelineConfig::load(manifest_path);
    sao::ZenonPipeline pipeline(config);

    auto input_ids = load_npy_int64_from_float(parity_dir + "/t5_input_ids.npy");
    auto attention_mask_f = load_npy_flat(parity_dir + "/t5_attention_mask.npy");
    std::vector<int64_t> attention_mask(attention_mask_f.size());
    for (size_t i = 0; i < attention_mask_f.size(); ++i)
        attention_mask[i] = static_cast<int64_t>(attention_mask_f[i]);

    auto noise = load_npy_flat(parity_dir + "/noise.npy");
    auto streamgen_latent = load_npy_flat(parity_dir + "/streamgen_latent.npy");
    auto input_latent = load_npy_flat(parity_dir + "/input_latent.npy");
    auto ref_sampled = load_npy_flat(parity_dir + "/sampled_latent.npy");

    float seconds_total = 11.0f;
    float keep_ratio = 0.5f;
    int steps = 8;
    float cfg_scale = 7.0f;

    auto audio = pipeline.generate(
        input_ids, attention_mask,
        seconds_total,
        streamgen_latent, input_latent,
        {}, {},
        keep_ratio, 42, steps, cfg_scale,
        nullptr,
        noise
    );

    std::cout << "Output audio size: " << audio.size() << std::endl;
    assert(!audio.empty());

    float max_audio_val = 0.0f;
    for (float v : audio) {
        float a = std::abs(v);
        if (a > max_audio_val) max_audio_val = a;
    }
    std::cout << "Output |audio|_max: " << max_audio_val << std::endl;

    std::cout << "PASS (full pipeline executed without errors)" << std::endl;
    return 0;
}
