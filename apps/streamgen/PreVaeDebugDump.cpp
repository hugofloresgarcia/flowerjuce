#include "PreVaeDebugDump.h"

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include <cassert>
#include <cstdlib>
#include <fstream>
#include <string>

namespace streamgen {

namespace {

const char* env_pre_vae_dir()
{
    return std::getenv("STREAMGEN_PRE_VAE_WAV_DIR");
}

bool write_row_major_stereo_float_wav(
    const juce::File& file,
    const std::vector<float>& streamgen_row_major_lr,
    int num_frames,
    double sample_rate_hz)
{
    assert(num_frames >= 1);
    assert(static_cast<int>(streamgen_row_major_lr.size()) == num_frames * 2);

    auto out = std::make_unique<juce::FileOutputStream>(file);
    if (!out->openedOk())
        return false;

    juce::WavAudioFormat wav;
    juce::AudioFormatWriterOptions opts;
    opts = opts.withSampleRate(sample_rate_hz)
              .withNumChannels(2)
              .withBitsPerSample(32)
              .withSampleFormat(juce::AudioFormatWriterOptions::SampleFormat::floatingPoint);

    std::unique_ptr<juce::OutputStream> stream = std::move(out);
    std::unique_ptr<juce::AudioFormatWriter> writer = wav.createWriterFor(stream, opts);
    if (writer == nullptr)
        return false;

    const float* L = streamgen_row_major_lr.data();
    const float* R = streamgen_row_major_lr.data() + static_cast<size_t>(num_frames);
    const float* channels[2] = {L, R};
    return writer->writeFromFloatArrays(channels, 2, num_frames);
}

void write_meta_file(
    const juce::File& file,
    const GenerationJob& job,
    double model_sample_rate_hz,
    int playback_sample_rate_hz,
    int num_frames,
    size_t streamgen_elems,
    size_t input_elems)
{
    std::ofstream out(file.getFullPathName().toStdString());
    if (!out)
        return;

    out << "STREAMGEN_PRE_VAE_WAV — tensors immediately before VAE encode (model-rate stereo).\n\n";
    out << "job_id=" << job.job_id << "\n";
    out << "model_sample_rate_hz=" << model_sample_rate_hz << "\n";
    out << "playback_sample_rate_hz=" << playback_sample_rate_hz << "\n";
    out << "window_start_sample=" << job.window_start_sample << "\n";
    out << "window_end_sample=" << job.window_end_sample << "\n";
    out << "window_length_samples_playback=" << job.window_length_samples() << "\n";
    out << "keep_end_sample=" << job.keep_end_sample << "\n";
    out << "keep_ratio=" << job.keep_ratio << "\n";
    out << "num_frames_stereo_model_rate=" << num_frames << "\n";
    out << "streamgen_float_count=" << streamgen_elems << "\n";
    out << "input_float_count=" << input_elems << "\n\n";
    out << "Alignment:\n"
        << "Both WAVs use the same playback-clock window [window_start, window_end) and resampled to\n"
        << "model_sample_rate_hz. Frame i is the same wall-clock instant in both files.\n"
        << "Drum input_audio is **silence** for abs_sample >= keep_end_sample (no future stem conditioning);\n"
        << "streamgen_audio uses the full window.\n\n"
        << "After ZenonPipeline::generate() completes, `pre_vae_<jobid>_zenon_output.wav` is written:\n"
        << "full VAE decode (2, num_frames) row-major at model rate (not the suffix-only slice written to the ring).\n";
}

} // namespace

void dump_pre_vae_wavs_if_enabled(
    const GenerationJob& job,
    double model_sample_rate_hz,
    int playback_sample_rate_hz,
    const std::vector<float>& streamgen_row_major_lr,
    const std::vector<float>& input_row_major_lr)
{
    const char* dir_c = env_pre_vae_dir();
    if (dir_c == nullptr || dir_c[0] == '\0')
        return;

    juce::File dir(juce::String::fromUTF8(dir_c));
    if (!dir.isDirectory())
        dir.createDirectory();

    const size_t sg = streamgen_row_major_lr.size();
    const size_t inp = input_row_major_lr.size();
    if (sg != inp)
    {
        DBG("StreamGen pre-VAE dump: size mismatch streamgen=" + juce::String(static_cast<int>(sg))
            + " input=" + juce::String(static_cast<int>(inp)) + " — skipping write");
        return;
    }
    if (sg < 2u || (sg % 2u) != 0u)
    {
        DBG("StreamGen pre-VAE dump: invalid stereo buffer size=" + juce::String(static_cast<int>(sg)));
        return;
    }

    const int num_frames = static_cast<int>(sg / 2u);

    const juce::String base = "pre_vae_" + juce::String(job.job_id)
        .paddedLeft('0', 8);

    const juce::File path_sg = dir.getChildFile(base + "_streamgen.wav");
    const juce::File path_in = dir.getChildFile(base + "_input.wav");
    const juce::File path_meta = dir.getChildFile(base + "_meta.txt");

    const bool ok_sg = write_row_major_stereo_float_wav(
        path_sg, streamgen_row_major_lr, num_frames, model_sample_rate_hz);
    const bool ok_in = write_row_major_stereo_float_wav(
        path_in, input_row_major_lr, num_frames, model_sample_rate_hz);

    write_meta_file(path_meta, job, model_sample_rate_hz, playback_sample_rate_hz,
                    num_frames, sg, inp);

    if (ok_sg && ok_in)
    {
        DBG("StreamGen pre-VAE dump: wrote " + path_sg.getFullPathName() + " "
            + path_in.getFullPathName() + " " + path_meta.getFullPathName());
    }
    else
    {
        DBG("StreamGen pre-VAE dump: write failed (streamgen_ok=" + juce::String(ok_sg ? 1 : 0)
            + " input_ok=" + juce::String(ok_in ? 1 : 0) + ") dir=" + dir.getFullPathName());
    }
}

void dump_zenon_pipeline_output_wav_if_enabled(
    const GenerationJob& job,
    double model_sample_rate_hz,
    int model_num_frames,
    const std::vector<float>& pipeline_output_row_major_lr)
{
    const char* dir_c = env_pre_vae_dir();
    if (dir_c == nullptr || dir_c[0] == '\0')
        return;

    if (model_num_frames < 1)
    {
        DBG("StreamGen zenon output dump: invalid model_num_frames");
        return;
    }

    const size_t expected = static_cast<size_t>(model_num_frames) * 2u;
    if (pipeline_output_row_major_lr.size() != expected)
    {
        DBG("StreamGen zenon output dump: size mismatch got=" + juce::String(static_cast<int>(pipeline_output_row_major_lr.size()))
            + " expected=" + juce::String(static_cast<int>(expected)) + " — skipping write");
        return;
    }

    juce::File dir(juce::String::fromUTF8(dir_c));
    if (!dir.isDirectory())
        dir.createDirectory();

    const juce::String base = "pre_vae_" + juce::String(job.job_id)
        .paddedLeft('0', 8);
    const juce::File path_out = dir.getChildFile(base + "_zenon_output.wav");

    const bool ok = write_row_major_stereo_float_wav(
        path_out, pipeline_output_row_major_lr, model_num_frames, model_sample_rate_hz);

    if (ok)
        DBG("StreamGen zenon output dump: wrote " + path_out.getFullPathName());
    else
        DBG("StreamGen zenon output dump: write failed path=" + path_out.getFullPathName());
}

} // namespace streamgen
