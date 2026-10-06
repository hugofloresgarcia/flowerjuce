// End-to-end parity: sao::Sa3Pipeline::generate vs the fp32 PyTorch reference run
// (tests/parity_data/pipeline/<arm>/<axis>/ from scripts/dump_sa3_reference_tensors.py).
//
// The pipeline gets the dumped ids/mask, raw control frames, context latent, keep ratio, lookahead, external noise,
// steps, cfg scale and axis, and is compared stage by stage:
//   conditioning (cross_attn, global, local_add, control, tf for both branches)      < 1e-4
//   per step: x_before, v_cond, v_uncond, v_cfg                                      reported (v: NN forward)
//   sampled latent                                                                   < 5e-3 (8-step loop)
//   decoded audio                                                                    reported as max abs + SNR; SNR > 40 dB
// Set SA3_PARITY_MIGRAPHX=1 to run every graph on the MIGraphX EP.
#include "sao_inference/Sa3Pipeline.h"
#include "sao_inference/Sa3PipelineConfig.h"

#include <cnpy.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> loadf(const fs::path& p)
{
    auto a = cnpy::npy_load(p.string());
    if (a.word_size != sizeof(float)) throw std::runtime_error(p.string() + ": not float32");
    if (a.fortran_order) throw std::runtime_error(p.string() + ": Fortran-ordered .npy (save C-contiguous arrays)");
    size_t n = 1;
    for (auto d : a.shape) n *= d;
    return std::vector<float>(a.data<float>(), a.data<float>() + n);
}

std::vector<int64_t> loadi(const fs::path& p)
{
    auto a = cnpy::npy_load(p.string());
    if (a.word_size != sizeof(int64_t)) throw std::runtime_error(p.string() + ": not int64");
    size_t n = 1;
    for (auto d : a.shape) n *= d;
    return std::vector<int64_t>(a.data<int64_t>(), a.data<int64_t>() + n);
}

float max_abs(const std::vector<float>& a, const std::vector<float>& b)
{
    if (a.size() != b.size()) return INFINITY;
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

double snr_db(const std::vector<float>& x, const std::vector<float>& ref)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += static_cast<double>(ref[i]) * ref[i];
        const double d = static_cast<double>(x[i]) - ref[i];
        den += d * d;
    }
    return 10.0 * std::log10(num / (den + 1e-20));
}

std::string step_name(int i, const char* what)
{
    std::ostringstream s;
    s << "step_" << std::setfill('0') << std::setw(2) << i << "_" << what << ".npy";
    return s.str();
}

} // namespace

int main()
{
    const fs::path root = SA3_REPO_ROOT;
    const fs::path data = fs::path(SA3_PARITY_DATA_DIR) / "pipeline";
    const bool mgx = std::getenv("SA3_PARITY_MIGRAPHX") != nullptr;
    bool ok = true;
    int tested = 0;
    for (const auto& arm_entry : fs::directory_iterator(data)) {
        const std::string arm = arm_entry.path().filename().string();
        const fs::path manifest = root / "models" / arm / "sa3_pipeline_manifest.json";
        if (!fs::exists(manifest)) {
            std::cout << "skip " << arm << ": no manifest" << std::endl;
            continue;
        }
        auto pc = sao::Sa3PipelineConfig::load(manifest.string());
        pc.use_migraphx = mgx;
        sao::Sa3Pipeline pipeline(pc);
        pipeline.set_verbose(false);
        for (const auto& axis_entry : fs::directory_iterator(arm_entry.path())) {
            const fs::path d = axis_entry.path();
            if (!fs::exists(d / "config.json")) continue;
            std::ifstream f(d / "config.json");
            const auto cfg = nlohmann::json::parse(f);
            std::cout << "=== pipeline parity " << arm << " / axis " << cfg.at("cfg_axis").get<std::string>() << (mgx ? " [MIGraphX]" : " [CPU]") << std::endl;

            sao::Sa3GenerateInputs in;
            in.input_ids = loadi(d / "input_ids.npy");
            in.attention_mask = loadi(d / "attention_mask.npy");
            in.control_frames = loadf(d / "control_raw.npy");
            in.context_latent = loadf(d / "context_latent.npy");
            in.keep_ratio = cfg.at("keep_ratio").get<float>();
            in.future_visibility_frames = cfg.at("future_visibility_frames").get<int>();
            in.steps = cfg.at("steps").get<int>();
            in.cfg_scale = cfg.at("cfg_scale").get<float>();
            in.cfg_axis = sao::parse_cfg_axis(cfg.at("cfg_axis").get<std::string>());
            in.external_noise = loadf(d / "noise.npy");

            float worst_v = 0.0f, worst_x = 0.0f;
            bool steps_ok = true;
            in.step_trace = [&](int i, float t, const std::vector<float>& x_before, const std::vector<float>& v_cond,
                                const std::vector<float>& v_uncond, const std::vector<float>& v_cfg) {
                const float ex = max_abs(x_before, loadf(d / step_name(i, "x_before")));
                const float ec = max_abs(v_cond, loadf(d / step_name(i, "v_cond")));
                const float eu = max_abs(v_uncond, loadf(d / step_name(i, "v_uncond")));
                const float eg = max_abs(v_cfg, loadf(d / step_name(i, "v_cfg")));
                const auto tref = loadf(d / step_name(i, "t"));
                std::cout << "  step " << i << " t=" << t << " (ref " << tref[0] << "): x " << ex << ", v_cond " << ec << ", v_uncond " << eu << ", v_cfg " << eg << std::endl;
                if (std::abs(t - tref[0]) > 1e-6f) steps_ok = false;
                worst_v = std::max({worst_v, ec, eu});
                worst_x = std::max(worst_x, ex);
            };
            const auto audio = pipeline.generate(in);

            const auto& cond = pipeline.last_cond_branch();
            const auto& uncond = pipeline.last_uncond_branch();
            const auto b_control = loadf(d / "branch_control.npy");      // [2, c, T]
            const auto b_local = loadf(d / "branch_local_add_cond.npy"); // [2, 257, T]
            const std::vector<float> ref_control_c(b_control.begin(), b_control.begin() + static_cast<std::ptrdiff_t>(cond.control.size()));
            const std::vector<float> ref_control_u(b_control.begin() + static_cast<std::ptrdiff_t>(cond.control.size()), b_control.end());
            const std::vector<float> ref_local_c(b_local.begin(), b_local.begin() + static_cast<std::ptrdiff_t>(cond.local_add_cond.size()));
            const std::vector<float> ref_local_u(b_local.begin() + static_cast<std::ptrdiff_t>(cond.local_add_cond.size()), b_local.end());
            const float e_cross = max_abs(cond.cross_attn_cond, loadf(d / "cross_attn_cond.npy"));
            const float e_global = max_abs(cond.global_embed, loadf(d / "global_embed.npy"));
            const float e_local = std::max(max_abs(cond.local_add_cond, ref_local_c), max_abs(uncond.local_add_cond, ref_local_u));
            const float e_control = std::max(max_abs(cond.control, ref_control_c), max_abs(uncond.control, ref_control_u));
            const float e_tf = max_abs(cond.tf_inpaint_mask, loadf(d / "tf_inpaint_mask.npy"));
            {   // where does a local_add mismatch live? channel 0 (mask) vs context channels, kept vs generated frames
                const int T = pc.window.frames;
                float e_mask = 0.0f, e_ctx_keep = 0.0f, e_ctx_rest = 0.0f;
                for (int t = 0; t < T; ++t) e_mask = std::max(e_mask, std::abs(cond.local_add_cond[static_cast<size_t>(t)] - ref_local_c[static_cast<size_t>(t)]));
                for (int c = 1; c < pc.dit.local_add_channels; ++c)
                    for (int t = 0; t < T; ++t) {
                        const size_t i = static_cast<size_t>(c) * T + t;
                        const float e = std::abs(cond.local_add_cond[i] - ref_local_c[i]);
                        if (t < pipeline.diagnostics().keep_frames) e_ctx_keep = std::max(e_ctx_keep, e); else e_ctx_rest = std::max(e_ctx_rest, e);
                    }
                std::cout << "  local_add detail: mask channel " << e_mask << ", context kept frames " << e_ctx_keep << ", context generated frames " << e_ctx_rest
                          << " (keep " << pipeline.diagnostics().keep_frames << ", tf " << pipeline.diagnostics().tf_keep_frames << ")" << std::endl;
            }
            std::cout << "  conditioning: cross_attn " << e_cross << " (text through ORT), global " << e_global << ", local_add " << e_local
                      << ", control " << e_control << ", tf " << e_tf << std::endl;
            const float e_latent = max_abs(pipeline.last_sampled_latent(), loadf(d / "sampled_latent.npy"));
            const auto ref_audio = loadf(d / "decoded_audio.npy");
            const float e_audio = max_abs(audio, ref_audio);
            const double snr = snr_db(audio, ref_audio);
            float peak = 0.0f;
            for (float v : ref_audio) peak = std::max(peak, std::abs(v));
            std::cout << "  sampled latent max abs " << e_latent << "; audio max abs " << e_audio << " (ref peak " << peak << "), SNR " << snr << " dB" << std::endl;
            const auto& tm = pipeline.timing();
            std::cout << "  timing: text " << tm.text_encode_ms << " ms, sampling " << tm.sampling_total_ms << " ms (" << in.steps << " steps), decode "
                      << tm.vae_decode_ms << " ms, total " << tm.total_ms << " ms" << std::endl;

            // The SAME-S decoder is chaotic on out-of-range latents (see EXPERIMENT.md, step 4): when the reference
            // audio itself exceeds full scale, the audio comparison says nothing about the C++ and only the latent counts.
            const bool audio_in_range = peak <= 1.0f;
            if (!audio_in_range) std::cout << "  (reference audio peaks at " << peak << " > 1: latent parity only)" << std::endl;
            const bool pass = steps_ok && e_global < 1e-4f && e_local < 1e-4f && e_control < 1e-4f && e_tf == 0.0f && e_cross < 1e-3f
                              && worst_v < 1e-2f && e_latent < 5e-3f && (!audio_in_range || snr > 40.0);
            std::cout << "  -> " << (pass ? "PASS" : "FAIL") << std::endl;
            ok = ok && pass;
            ++tested;
        }
    }
    if (tested == 0) {
        std::cerr << "no pipeline parity data under " << data << std::endl;
        return 1;
    }
    std::cout << (ok ? "PASS" : "FAIL") << " (" << tested << " cases)" << std::endl;
    return ok ? 0 : 1;
}
