// Parity test: sao::NumberEmbedder (expo Fourier features) vs the SA3 `seconds_total` conditioner.
// Data: tests/parity_data/number_embedder/{config.json (weights_dir), values.npy [N], embeds_ref.npy [N, 768]}
// written by scripts/export_sa3_number_embedder.py. Threshold: pure math, 1e-5 max abs.
#include "sao_inference/NumberEmbedder.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

int main()
{
    const std::string dir = std::string(SA3_PARITY_DATA_DIR) + "/number_embedder";
    std::ifstream f(dir + "/config.json");
    if (!f) {
        std::cerr << "cannot open " << dir << "/config.json" << std::endl;
        return 1;
    }
    const auto cfg = nlohmann::json::parse(f);
    const std::string weights_dir = std::string(SA3_REPO_ROOT) + "/" + cfg.at("weights_dir").get<std::string>();
    std::cout << "=== SA3 number embedder parity: " << weights_dir << " ===" << std::endl;

    sao::NumberEmbedder embedder(weights_dir);
    auto values = cnpy::npy_load(dir + "/values.npy");
    auto ref = cnpy::npy_load(dir + "/embeds_ref.npy");
    const size_t n = values.shape[0];
    const size_t dim = ref.shape[1];
    if (ref.shape[0] != n || static_cast<int>(dim) != embedder.features()) {
        std::cerr << "shape mismatch: values " << n << ", ref " << ref.shape[0] << " x " << dim << ", features " << embedder.features() << std::endl;
        return 1;
    }
    const float* vals = values.data<float>();
    const float* r = ref.data<float>();
    float worst = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const auto e = embedder.embed(vals[i]);
        float err = 0.0f;
        for (size_t k = 0; k < dim; ++k) err = std::max(err, std::abs(e[k] - r[i * dim + k]));
        std::cout << "value " << vals[i] << ": max abs err " << err << std::endl;
        worst = std::max(worst, err);
    }
    constexpr float kThreshold = 1e-5f;
    std::cout << "worst " << worst << (worst < kThreshold ? " PASS" : " FAIL") << " (threshold " << kThreshold << ")" << std::endl;
    return worst < kThreshold ? 0 : 1;
}
