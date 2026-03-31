/// Regression: loop-hold must not play last-gen snapshot before its scheduled land time.
///
/// Build: `cmake --build --preset debug-mlx --target StreamGenProcessorHoldTests -j`
/// Run: `flowerjuce/build/apps/streamgen/StreamGenProcessorHoldTests`

#include "StreamGenProcessor.h"
#include "TimeRuler.h"

#include <cmath>
#include <cstdlib>
#include <vector>

int main()
{
    streamgen::StreamGenProcessor processor;
    streamgen::ModelConstants constants;
    constants.sample_rate = 44100;
    constants.sample_size = 512;
    processor.configure(constants);
    processor.loop_last_generation.store(true, std::memory_order_relaxed);
    processor.scheduler().generation_enabled.store(true, std::memory_order_relaxed);

    const int64_t land_start = 100000;
    const int64_t num_samples = 512;
    std::vector<float> gen_row_major(static_cast<size_t>(num_samples * streamgen::StreamGenProcessor::NUM_CHANNELS));
    for (int64_t i = 0; i < num_samples; ++i)
    {
        gen_row_major[static_cast<size_t>(i)] = 0.9f;
        gen_row_major[static_cast<size_t>(static_cast<size_t>(num_samples) + static_cast<size_t>(i))] = 0.9f;
    }

    processor.write_output(gen_row_major, land_start, num_samples, 0);

    const streamgen::DrumsRingSample before_land = processor.fetch_drums_ring_sample(land_start - 1);
    if (before_land.from_last_gen_hold)
        return 1;
    if (std::fabs(before_land.left) > 1.0e-5f || std::fabs(before_land.right) > 1.0e-5f)
        return 2;

    const streamgen::DrumsRingSample in_land = processor.fetch_drums_ring_sample(land_start + 100);
    if (in_land.from_last_gen_hold)
        return 3;
    if (std::fabs(in_land.left - 0.9f) > 1.0e-5f || std::fabs(in_land.right - 0.9f) > 1.0e-5f)
        return 4;

    const streamgen::DrumsRingSample looped = processor.fetch_drums_ring_sample(land_start + num_samples + 7);
    if (!looped.from_last_gen_hold)
        return 5;
    if (std::fabs(looped.left - gen_row_major[static_cast<size_t>(7)]) > 1.0e-5f)
        return 6;

    return 0;
}
