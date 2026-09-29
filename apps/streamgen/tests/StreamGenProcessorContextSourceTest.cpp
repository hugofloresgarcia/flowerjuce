/// Per-region drum context: frames flagged live must come from the percussion ring, the rest from
/// the drums output ring, and nothing past keep_end.
///
/// Build: `cmake --build --preset debug --target StreamGenProcessorContextSourceTests -j`
/// Run: `flowerjuce/build/apps/streamgen/StreamGenProcessorContextSourceTests`

#include "StreamGenProcessor.h"
#include "TimeRuler.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr float k_tol = 1.0e-5f;

bool near(float a, float b)
{
    return std::fabs(a - b) <= k_tol;
}

} // namespace

int main()
{
    streamgen::StreamGenProcessor processor;
    streamgen::ModelConstants constants;
    constants.sample_rate = 44100;
    constants.sample_size = 1024;
    processor.configure(constants);
    // Playback rate equals model rate here, so the VAE snapshot is not resampled.

    constexpr int64_t window_start = 0;
    constexpr int64_t keep_end = 512;
    constexpr int64_t window_len = 1024;
    constexpr int64_t live_from = 256;
    constexpr float drums_val = 0.9f;
    constexpr float perc_self_val = 0.3f; // recorded while the toggle was off: must be ignored
    constexpr float perc_live_val = 0.5f;

    // Feed the first 256 frames with the toggle off, then the rest with it on.
    constexpr int block = 64;
    std::vector<float> sax(static_cast<size_t>(block), 0.0f);
    std::vector<float> perc(static_cast<size_t>(block), 0.0f);
    for (int64_t pos = 0; pos < window_len; pos += block)
    {
        const bool live = pos >= live_from;
        processor.percussion_as_context.store(live, std::memory_order_relaxed);
        std::fill(perc.begin(), perc.end(), live ? perc_live_val : perc_self_val);
        processor.feed_percussion_audio(perc.data(), block);
        processor.feed_streamgen_audio(sax.data(), block);
    }

    // Model output covering the whole kept prefix.
    std::vector<float> gen(static_cast<size_t>(keep_end * streamgen::StreamGenProcessor::NUM_CHANNELS), drums_val);
    processor.write_output(gen, window_start, keep_end, 0);

    if (processor.count_live_context_samples(window_start, keep_end) != keep_end - live_from)
        return 1;

    const std::vector<float> input_audio =
        processor.snapshot_input_audio_for_vae(window_start, window_len, keep_end);
    if (static_cast<int64_t>(input_audio.size()) != window_len * streamgen::StreamGenProcessor::NUM_CHANNELS)
        return 2;

    for (int64_t i = 0; i < window_len; ++i)
    {
        const float L = input_audio[static_cast<size_t>(i)];
        const float R = input_audio[static_cast<size_t>(window_len + i)];
        float expected = 0.0f;
        if (i < live_from)
            expected = drums_val; // self region: model's own output
        else if (i < keep_end)
            expected = perc_live_val; // live region: percussion ring, mono duplicated
        if (!near(L, expected) || !near(R, expected))
            return 3;
    }

    // Live silence must stay silence even though a warmup stem would otherwise fill silent frames.
    std::fill(perc.begin(), perc.end(), 0.0f);
    processor.percussion_as_context.store(true, std::memory_order_relaxed);
    processor.feed_percussion_audio(perc.data(), block);
    processor.feed_streamgen_audio(sax.data(), block);
    const int64_t silent_start = window_len;
    const std::vector<float> silent = processor.snapshot_input_audio_for_vae(silent_start, block, silent_start + block);
    for (float v : silent)
        if (!near(v, 0.0f))
            return 4;

    return 0;
}
