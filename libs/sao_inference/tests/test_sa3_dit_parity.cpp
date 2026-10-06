// Parity test: sao::Sa3DiTModel vs fp32 PyTorch on the dumped two-branch conditioning
// (tests/parity_data/dit/<arm>/ from scripts/export_sa3_dit.py), for every arm that has a manifest.
// Threshold: 1e-3 max abs on v (ORT CPU measured ~5e-5, MIGraphX ~1e-4). Set SA3_PARITY_MIGRAPHX=1 for the GPU EP.
#include "sao_inference/Sa3DiTModel.h"
#include "sao_inference/Sa3PipelineConfig.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> load(const fs::path& p)
{
    auto a = cnpy::npy_load(p.string());
    if (a.word_size != sizeof(float)) throw std::runtime_error(p.string() + ": not float32");
    if (a.fortran_order) throw std::runtime_error(p.string() + ": Fortran-ordered .npy (save C-contiguous arrays)");
    size_t n = 1;
    for (auto d : a.shape) n *= d;
    return std::vector<float>(a.data<float>(), a.data<float>() + n);
}

} // namespace

int main()
{
    const fs::path root = SA3_REPO_ROOT;
    const fs::path dit_data = fs::path(SA3_PARITY_DATA_DIR) / "dit";
    const bool mgx = std::getenv("SA3_PARITY_MIGRAPHX") != nullptr;
    constexpr float kThreshold = 1e-3f;
    bool ok = true;
    int tested = 0;
    for (const auto& entry : fs::directory_iterator(dit_data)) {
        const fs::path d = entry.path();
        if (!fs::exists(d / "config.json")) continue;
        std::ifstream f(d / "config.json");
        const auto cfg = nlohmann::json::parse(f);
        const std::string arm = cfg.at("arm").get<std::string>();
        const fs::path manifest = root / "models" / arm / "sa3_pipeline_manifest.json";
        if (!fs::exists(manifest)) {
            std::cout << "skip " << arm << ": no manifest" << std::endl;
            continue;
        }
        std::cout << "=== DiT parity " << arm << (mgx ? " [MIGraphX]" : " [CPU]") << std::endl;
        auto pc = sao::Sa3PipelineConfig::load(manifest.string());
        pc.use_migraphx = mgx;
        sao::Sa3DiTModel dit(pc.dit_onnx(pc.dit.batch), pc.dit.inputs, pc.dit.output, pc.dit.migraphx_cache_dir, false, false, mgx);
        sao::Sa3DiTModel::Dims dims;
        dims.batch = cfg.at("batch").get<int>();
        dims.channels = cfg.at("io_channels").get<int>();
        dims.frames = cfg.at("frames").get<int>();
        dims.cross_tokens = pc.dit.cross_attn_tokens;
        dims.cond_dim = pc.dit.cond_token_dim;
        dims.global_dim = pc.dit.global_cond_dim;
        dims.local_add_channels = pc.dit.local_add_channels;
        dims.control_channels = cfg.at("control_dim").get<int>();
        const auto cross = load(d / "cross_attn_cond.npy");
        const auto global = load(d / "global_embed.npy");
        const auto local = load(d / "local_add_cond.npy");
        const auto control = load(d / "control.npy");
        const auto tf = load(d / "tf_inpaint_mask.npy");
        const int n_t = cfg.at("n_timesteps").get<int>();
        float worst = 0.0f;
        for (int i = 0; i < n_t; ++i) {
            const auto x = load(d / ("x_" + std::to_string(i) + ".npy"));
            const auto t = load(d / ("t_" + std::to_string(i) + ".npy"));
            const auto ref = load(d / ("v_ref_" + std::to_string(i) + ".npy"));
            const auto v = dit.forward(x, t, cross, global, local, control, tf, dims);
            float err = 0.0f;
            for (size_t k = 0; k < v.size(); ++k) err = std::max(err, std::abs(v[k] - ref[k]));
            std::cout << "  t=" << t[0] << ": max abs err " << err << std::endl;
            worst = std::max(worst, err);
        }
        if (!(worst < kThreshold)) ok = false;
        ++tested;
    }
    if (tested == 0) {
        std::cerr << "no DiT parity data found under " << dit_data << std::endl;
        return 1;
    }
    std::cout << (ok ? "PASS" : "FAIL") << " (" << tested << " arms, threshold " << kThreshold << ")" << std::endl;
    return ok ? 0 : 1;
}
