// Parity test: sao::Tokenizer + the T5Gemma ONNX encoder vs the SA3 prompt conditioner.
// Data: tests/parity_data/text/{prompts.json, input_ids.npy [P,L], attention_mask.npy [P,L], embeddings_ref.npy [P,L,768],
//       config.json (onnx path, tokenizer options)} from scripts/export_sa3_text_encoder.py.
// Checks: token ids and mask identical; embeddings max abs < 1e-3 (NN forward through ORT).
// Set SA3_PARITY_MIGRAPHX=1 (and ORT_MIGRAPHX_MODEL_CACHE_PATH) to run the graph on the MIGraphX EP instead of CPU.
#include "sao_inference/T5Encoder.h"
#include "sao_inference/Tokenizer.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main()
{
    const std::string dir = std::string(SA3_PARITY_DATA_DIR) + "/text";
    const std::string root = SA3_REPO_ROOT;
    std::ifstream cf(dir + "/config.json");
    if (!cf) {
        std::cerr << "cannot open " << dir << "/config.json" << std::endl;
        return 1;
    }
    const auto cfg = nlohmann::json::parse(cf);
    std::ifstream pf(dir + "/prompts.json");
    const std::vector<std::string> prompts = nlohmann::json::parse(pf).get<std::vector<std::string>>();
    const std::string onnx_path = root + "/" + cfg.at("onnx").get<std::string>();
    const auto& tk = cfg.at("tokenizer");
    const std::string tokenizer_path = onnx_path.substr(0, onnx_path.find_last_of('/')) + "/" + tk.at("model_file").get<std::string>();
    const bool use_migraphx = std::getenv("SA3_PARITY_MIGRAPHX") != nullptr;
    std::cout << "=== SA3 text encoder parity: " << onnx_path << (use_migraphx ? " [MIGraphX]" : " [CPU]") << " ===" << std::endl;

    sao::TokenizerOptions opts;
    opts.max_length = tk.at("max_length").get<int>();
    opts.add_bos = tk.at("add_bos").get<bool>();
    opts.add_eos = tk.at("add_eos").get<bool>();
    opts.pad_id = tk.at("pad_id").get<int>();
    opts.bos_id = tk.at("bos_id").get<int>();
    opts.eos_id = tk.at("eos_id").get<int>();
    sao::Tokenizer tokenizer(tokenizer_path, opts);

    auto ids_ref = cnpy::npy_load(dir + "/input_ids.npy");
    auto mask_ref = cnpy::npy_load(dir + "/attention_mask.npy");
    auto emb_ref = cnpy::npy_load(dir + "/embeddings_ref.npy");
    const size_t P = ids_ref.shape[0];
    const size_t L = ids_ref.shape[1];
    const size_t D = emb_ref.shape[2];
    if (P != prompts.size() || static_cast<int>(L) != opts.max_length || ids_ref.word_size != 8 || mask_ref.word_size != 8) {
        std::cerr << "unexpected parity data layout" << std::endl;
        return 1;
    }

    bool ok = true;
    for (size_t p = 0; p < P; ++p) {
        const auto t = tokenizer.tokenize(prompts[p]);
        const int64_t* ids = ids_ref.data<int64_t>() + p * L;
        const int64_t* mask = mask_ref.data<int64_t>() + p * L;
        size_t bad = 0;
        for (size_t i = 0; i < L; ++i) bad += (t.input_ids[i] != ids[i]) + (t.attention_mask[i] != mask[i]);
        int64_t n = 0;
        for (size_t i = 0; i < L; ++i) n += mask[i];
        std::cout << "prompt " << p << " (" << n << " tokens): id/mask mismatches " << bad << std::endl;
        if (bad) ok = false;
    }

    sao::T5Encoder encoder(onnx_path, false, false, use_migraphx);
    float worst = 0.0f;
    for (size_t p = 0; p < P; ++p) {
        std::vector<int64_t> ids(ids_ref.data<int64_t>() + p * L, ids_ref.data<int64_t>() + (p + 1) * L);
        std::vector<int64_t> mask(mask_ref.data<int64_t>() + p * L, mask_ref.data<int64_t>() + (p + 1) * L);
        const auto emb = encoder.encode(ids, mask);
        if (emb.size() != L * D) {
            std::cerr << "embedding size " << emb.size() << " != " << L * D << std::endl;
            return 1;
        }
        const float* ref = emb_ref.data<float>() + p * L * D;
        float err = 0.0f;
        for (size_t i = 0; i < L * D; ++i) {
            if (!std::isfinite(emb[i])) err = INFINITY;
            err = std::max(err, std::abs(emb[i] - ref[i]));
        }
        std::cout << "prompt " << p << ": embeddings max abs err " << err << std::endl;
        worst = std::max(worst, err);
    }
    constexpr float kThreshold = 1e-3f;
    if (!(worst < kThreshold)) ok = false;
    std::cout << "worst embedding error " << worst << " (threshold " << kThreshold << ") -> " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok ? 0 : 1;
}
