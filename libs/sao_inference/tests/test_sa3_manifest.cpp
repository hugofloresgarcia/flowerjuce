// Smoke test for sao::Sa3PipelineConfig: load every models/*/sa3_pipeline_manifest.json under the repo,
// then build the manifest-driven CPU components from it (Tokenizer, NumberEmbedder, ControlFeatureExtractor)
// and run each once. No ONNX session is created here (the pipeline test does that).
#include "sao_inference/ControlFeatureExtractor.h"
#include "sao_inference/NumberEmbedder.h"
#include "sao_inference/Sa3PipelineConfig.h"
#include "sao_inference/Tokenizer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main()
{
    const fs::path models = fs::path(SA3_REPO_ROOT) / "models";
    std::vector<fs::path> manifests;
    for (const auto& entry : fs::directory_iterator(models)) {
        const fs::path m = entry.path() / "sa3_pipeline_manifest.json";
        if (fs::exists(m)) manifests.push_back(m);
    }
    if (manifests.empty()) {
        std::cerr << "no sa3_pipeline_manifest.json under " << models << std::endl;
        return 1;
    }
    bool ok = true;
    for (const auto& m : manifests) {
        std::cout << "=== " << m << std::endl;
        try {
            const auto cfg = sao::Sa3PipelineConfig::load(m.string());
            std::cout << "arm " << cfg.arm << " step " << cfg.global_step << ": window " << cfg.window.frames << " fr / "
                      << cfg.window.seconds_total << " s, control " << cfg.control.id << " dim " << cfg.control.dim
                      << " null " << cfg.control.null_value << ", dit batch " << cfg.dit.batch << " -> " << cfg.dit_onnx(cfg.dit.batch)
                      << ", cfg " << cfg.cfg.mode << " axis " << cfg.cfg.axis_default << " x" << cfg.cfg.scale_default
                      << ", sampler " << cfg.sampler.type << " " << cfg.sampler.steps_default << " steps" << std::endl;

            sao::Tokenizer tokenizer(cfg.text.tokenizer_model, cfg.text.tokenizer);
            const auto tok = tokenizer.tokenize("saxophone");
            int n = 0;
            for (auto v : tok.attention_mask) n += static_cast<int>(v);
            if (static_cast<int>(tok.input_ids.size()) != cfg.text.max_length || n != 2) {
                std::cerr << "tokenizer: unexpected result for 'saxophone' (" << n << " tokens)" << std::endl;
                ok = false;
            }

            sao::NumberEmbedder number(cfg.number_embedder.weights_dir);
            const auto emb = number.embed(cfg.window.seconds_total);
            if (static_cast<int>(emb.size()) != cfg.dit.global_cond_dim || !std::isfinite(emb[0])) {
                std::cerr << "number embedder: bad output size " << emb.size() << std::endl;
                ok = false;
            }

            sao::ControlFeatureExtractor features(cfg.control.feature);
            std::vector<float> silence(static_cast<size_t>(cfg.downsampling_ratio) * 3, 0.0f);
            features.push_mono(silence.data(), silence.size());
            if (features.frames_available() != 3 || features.channels() != cfg.control.dim) {
                std::cerr << "feature extractor: " << features.frames_available() << " frames, " << features.channels() << " channels" << std::endl;
                ok = false;
            }
            std::vector<float> frame(static_cast<size_t>(features.channels()));
            features.pop_frame(frame.data());
            std::cout << "silence frame: ";
            for (float v : frame) std::cout << v << " ";
            std::cout << "(" << features.channels() << " ch)" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "FAIL: " << e.what() << std::endl;
            ok = false;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
