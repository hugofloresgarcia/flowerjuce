#include "sao_inference/InpaintSampler.h"
#include "sao_inference/RectifiedFlowSampler.h"
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

int main()
{
    std::string parity_dir = ZENON_PARITY_DATA_DIR;

    std::cout << "=== Zenon Sampler Parity Test (distribution shift) ===" << std::endl;

    auto ref_before = load_npy_flat(parity_dir + "/t_schedule_before_shift.npy");
    auto ref_after = load_npy_flat(parity_dir + "/t_schedule_after_shift.npy");

    int steps = static_cast<int>(ref_before.size()) - 1;
    auto schedule = sao::build_time_schedule(steps, 1.0f);

    std::cout << "Steps: " << steps << std::endl;
    assert(schedule.size() == ref_before.size());

    float pre_shift_err = 0.0f;
    for (size_t i = 0; i < schedule.size(); ++i) {
        float err = std::abs(schedule[i] - ref_before[i]);
        if (err > pre_shift_err) pre_shift_err = err;
    }
    std::cout << "Pre-shift max error: " << pre_shift_err << std::endl;

    sao::DistributionShiftConfig ds;
    ds.enabled = true;
    ds.base_shift = 0.5f;
    ds.max_shift = 1.15f;
    ds.min_length = 256;
    ds.max_length = 4096;

    sao::apply_distribution_shift(schedule, ds, 256);

    float post_shift_err = 0.0f;
    for (size_t i = 0; i < schedule.size(); ++i) {
        float err = std::abs(schedule[i] - ref_after[i]);
        if (err > post_shift_err) post_shift_err = err;
        if (err > 1e-3f) {
            std::cout << "  Step " << i << ": got=" << schedule[i]
                      << " ref=" << ref_after[i] << " err=" << err << std::endl;
        }
    }
    std::cout << "Post-shift max error: " << post_shift_err << std::endl;

    constexpr float THRESHOLD = 1e-4f;
    bool pass = pre_shift_err < THRESHOLD && post_shift_err < THRESHOLD;

    if (pass) {
        std::cout << "PASS (threshold=" << THRESHOLD << ")" << std::endl;
        return 0;
    } else {
        std::cerr << "FAIL" << std::endl;
        return 1;
    }
}
