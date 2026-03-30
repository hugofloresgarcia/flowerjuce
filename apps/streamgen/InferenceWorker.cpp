#include "InferenceWorker.h"

#include <chrono>

namespace streamgen {

InferenceWorker::InferenceWorker(StreamGenProcessor& processor)
    : juce::Thread("StreamGen Inference Worker"),
      m_processor(processor)
{
}

InferenceWorker::~InferenceWorker()
{
    stopThread(5000);
}

bool InferenceWorker::load_pipeline(const std::string& manifest_path, bool use_cuda, bool use_coreml)
{
    DBG("InferenceWorker: loading pipeline from " + juce::String(manifest_path));

    m_config = sao::ZenonPipelineConfig::load(manifest_path);
    m_config.use_cuda = use_cuda;
    m_config.use_coreml = use_coreml;

    m_pipeline = std::make_unique<sao::ZenonPipeline>(m_config);

    // Load tokenizer from same directory as t5 onnx
    std::string t5_dir = m_config.t5_onnx_path;
    auto last_slash = t5_dir.rfind('/');
    std::string tokenizer_dir = (last_slash != std::string::npos) ? t5_dir.substr(0, last_slash) : ".";
    std::string sp_path = tokenizer_dir + "/t5_tokenizer.model";

    m_tokenizer = std::make_unique<sao::Tokenizer>(sp_path, 64);

    ModelConstants constants;
    constants.sample_rate = m_config.sample_rate;
    constants.sample_size = m_config.sample_size;
    constants.latent_dim = m_config.latent_dim;
    constants.latent_length = m_config.latent_length;
    constants.downsampling_ratio = m_config.downsampling_ratio;
    m_processor.configure(constants);

    DBG("InferenceWorker: pipeline loaded successfully"
        " (sample_size=" + juce::String(m_config.sample_size)
        + ", latent_dim=" + juce::String(m_config.latent_dim)
        + ", cuda=" + juce::String(use_cuda ? "yes" : "no") + ")");

    return true;
}

void InferenceWorker::set_prompt(const std::string& prompt)
{
    std::lock_guard<std::mutex> lock(m_prompt_mutex);
    if (m_prompt != prompt)
    {
        m_prompt = prompt;
        m_prompt_dirty = true;
        DBG("InferenceWorker: prompt changed to '" + juce::String(prompt) + "'");
    }
}

StageTiming InferenceWorker::last_timing() const
{
    std::lock_guard<std::mutex> lock(m_timing_mutex);
    return m_last_timing;
}

void InferenceWorker::run()
{
    DBG("InferenceWorker: thread started");

    while (!threadShouldExit())
    {
        GenerationJob job;
        bool have_job = m_processor.scheduler().pop_job(job);

        if (!have_job)
        {
            sleep(10);
            continue;
        }

        m_processor.scheduler().status.worker_busy.store(true, std::memory_order_relaxed);
        m_processor.scheduler().status.last_job_id.store(job.job_id, std::memory_order_relaxed);

        process_job(job);

        m_processor.scheduler().status.worker_busy.store(false, std::memory_order_relaxed);
        m_processor.scheduler().status.generation_count.fetch_add(1, std::memory_order_relaxed);
    }

    DBG("InferenceWorker: thread exiting");
}

void InferenceWorker::process_job(const GenerationJob& job)
{
    DBG("InferenceWorker: processing job #" + juce::String(job.job_id)
        + " window=[" + juce::String(job.window_start_sample)
        + ", " + juce::String(job.window_end_sample) + "]"
        + " keep_ratio=" + juce::String(job.keep_ratio, 3));

    auto t_start = std::chrono::steady_clock::now();

    // Ensure tokenization is up to date
    update_tokenization();

    // Snapshot sax audio (streamgen_audio) from input ring buffer
    auto sax_audio = m_processor.snapshot_input(
        job.window_start_sample, job.window_length_samples());

    // Snapshot previous drums (input_audio) from output ring buffer
    auto drums_audio = m_processor.snapshot_output(
        job.window_start_sample, job.window_length_samples());

    // Run the pipeline
    std::vector<float> empty_latent;
    auto output = m_pipeline->generate(
        m_input_ids,
        m_attention_mask,
        job.seconds_total,
        empty_latent,    // streamgen_latent (empty -> will VAE encode sax_audio)
        empty_latent,    // input_latent (empty -> will VAE encode drums_audio)
        sax_audio,       // streamgen_audio
        drums_audio,     // input_audio
        job.keep_ratio,
        static_cast<uint32_t>(job.job_id), // use job_id as seed for variety
        job.steps,
        job.cfg_scale
    );

    auto t_end = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    // Extract the generated portion (skip the kept prefix in the decoded audio)
    int64_t model_samples = m_config.sample_size;
    int64_t keep_samples = job.keep_end_sample - job.window_start_sample;
    int64_t gen_samples = model_samples - keep_samples;

    assert(static_cast<int64_t>(output.size()) >= model_samples * 2);

    // output is (2, model_samples) row-major
    // Extract [keep_samples, model_samples) for both channels
    std::vector<float> gen_audio(static_cast<size_t>(gen_samples) * 2);
    for (int64_t i = 0; i < gen_samples; ++i)
    {
        gen_audio[static_cast<size_t>(i)] = output[static_cast<size_t>(keep_samples + i)];
        gen_audio[static_cast<size_t>(gen_samples + i)] = output[static_cast<size_t>(model_samples + keep_samples + i)];
    }

    // Write to output ring buffer with crossfade
    int xfade = crossfade_samples.load(std::memory_order_relaxed);
    m_processor.write_output(gen_audio, job.output_start_sample(), gen_samples, xfade);

    // Update timing
    const auto& pipeline_timing = m_pipeline->timing();
    {
        std::lock_guard<std::mutex> lock(m_timing_mutex);
        m_last_timing.vae_encode_ms = pipeline_timing.vae_encode_ms;
        m_last_timing.t5_encode_ms = pipeline_timing.t5_encode_ms;
        m_last_timing.sampling_total_ms = pipeline_timing.sampling_total_ms;
        m_last_timing.vae_decode_ms = pipeline_timing.vae_decode_ms;
        m_last_timing.total_ms = total_ms;
        m_last_timing.steps = job.steps;
    }

    m_processor.scheduler().status.last_latency_ms.store(total_ms, std::memory_order_relaxed);

    DBG("InferenceWorker: job #" + juce::String(job.job_id)
        + " completed in " + juce::String(total_ms, 1) + "ms"
        + " (vae_enc=" + juce::String(pipeline_timing.vae_encode_ms, 1)
        + " t5=" + juce::String(pipeline_timing.t5_encode_ms, 1)
        + " dit=" + juce::String(pipeline_timing.sampling_total_ms, 1)
        + " vae_dec=" + juce::String(pipeline_timing.vae_decode_ms, 1) + ")");
}

void InferenceWorker::update_tokenization()
{
    std::lock_guard<std::mutex> lock(m_prompt_mutex);

    if (!m_prompt_dirty)
        return;

    assert(m_tokenizer != nullptr);
    auto result = m_tokenizer->tokenize(m_prompt);
    m_input_ids = result.input_ids;
    m_attention_mask = result.attention_mask;
    m_prompt_dirty = false;

    DBG("InferenceWorker: re-tokenized prompt '" + juce::String(m_prompt)
        + "' -> " + juce::String(static_cast<int>(m_input_ids.size())) + " tokens");
}

} // namespace streamgen
