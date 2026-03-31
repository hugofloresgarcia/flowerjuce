#include "InferenceWorker.h"
#include "StreamGenDebugLog.h"
#include "GenerationTimelineStore.h"

#include <algorithm>
#include <chrono>

namespace streamgen {

InferenceWorker::InferenceWorker(StreamGenProcessor& processor)
    : juce::Thread("StreamGen Inference Worker"),
      m_processor(processor),
      m_timeline(processor.timeline_store())
{
}

InferenceWorker::~InferenceWorker()
{
    stopThread(5000);
}

bool InferenceWorker::load_pipeline(
    const std::string& manifest_path,
    bool use_cuda,
    bool use_coreml,
    bool use_mlx_vae)
{
    DBG("InferenceWorker: loading pipeline from " + juce::String(manifest_path));
    streamgen_log("InferenceWorker::load_pipeline begin: " + juce::String(manifest_path));

    m_config = sao::ZenonPipelineConfig::load(manifest_path);
    m_config.use_cuda = use_cuda;
    m_config.use_coreml = use_coreml;
    m_config.use_mlx_vae = use_mlx_vae;

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

    m_pipeline->set_verbose(m_pipeline_verbose);

    m_have_t5_cache = false;
    m_cached_t5_masked.clear();
    m_t5_cache_ids.clear();
    m_t5_cache_mask.clear();

    DBG("InferenceWorker: pipeline loaded successfully"
        " (sample_size=" + juce::String(m_config.sample_size)
        + ", latent_dim=" + juce::String(m_config.latent_dim)
        + ", cuda=" + juce::String(use_cuda ? "yes" : "no") + ")");
    streamgen_log("InferenceWorker::load_pipeline OK sample_size=" + juce::String(m_config.sample_size)
        + " model_sr=" + juce::String(m_config.sample_rate)
        + " processor_abs_pos=" + juce::String(m_processor.scheduler().absolute_sample_pos()));

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
    std::lock_guard<std::mutex> lock(m_worker_state_mutex);
    return m_last_timing;
}

InferenceSnapshot InferenceWorker::last_snapshot() const
{
    std::lock_guard<std::mutex> lock(m_worker_state_mutex);
    return m_last_snapshot;
}

void InferenceWorker::set_pipeline_verbose(bool verbose)
{
    m_pipeline_verbose = verbose;
    if (m_pipeline != nullptr)
        m_pipeline->set_verbose(verbose);
}

void InferenceWorker::run()
{
    DBG("InferenceWorker: thread started");
    streamgen_log("InferenceWorker::run thread started");

    int idle_spins = 0;
    while (!threadShouldExit())
    {
        GenerationJob job;
        bool have_job = m_processor.scheduler().pop_job(job);

        if (!have_job)
        {
            ++idle_spins;
            if (idle_spins <= 10 || idle_spins % 500 == 0)
                streamgen_log("InferenceWorker: idle (no job) spin=" + juce::String(idle_spins)
                    + " abs_pos=" + juce::String(m_processor.scheduler().absolute_sample_pos()));
            sleep(10);
            continue;
        }

        idle_spins = 0;
        m_processor.scheduler().status.worker_busy.store(true, std::memory_order_relaxed);
        m_processor.scheduler().status.last_job_id.store(job.job_id, std::memory_order_relaxed);

        process_job(job);

        m_processor.scheduler().status.worker_busy.store(false, std::memory_order_relaxed);
        m_processor.scheduler().status.generation_count.fetch_add(1, std::memory_order_relaxed);
    }

    DBG("InferenceWorker: thread exiting");
    streamgen_log("InferenceWorker::run thread exiting");
}

void InferenceWorker::process_job(const GenerationJob& job)
{
    DBG("InferenceWorker: processing job #" + juce::String(job.job_id)
        + " window=[" + juce::String(job.window_start_sample)
        + ", " + juce::String(job.window_end_sample) + "]"
        + " keep_ratio=" + juce::String(job.keep_ratio, 3));

    streamgen_log("InferenceWorker::process_job START id=" + juce::String(job.job_id)
        + " win=[" + juce::String(job.window_start_sample) + "," + juce::String(job.window_end_sample)
        + ") keep_end=" + juce::String(job.keep_end_sample)
        + " out_start=" + juce::String(job.output_start_sample())
        + " abs_pos_now=" + juce::String(m_processor.scheduler().absolute_sample_pos()));

    auto t_start = std::chrono::steady_clock::now();

    // Ensure tokenization is up to date
    update_tokenization();

    constexpr int k_t5_seq_len = 64;
    const int t5_masked_elems = k_t5_seq_len * m_config.cond_token_dim;
    const bool reuse_t5 = m_have_t5_cache
        && static_cast<int>(m_cached_t5_masked.size()) == t5_masked_elems
        && m_input_ids.size() == m_t5_cache_ids.size()
        && m_attention_mask.size() == m_t5_cache_mask.size()
        && std::equal(m_input_ids.begin(), m_input_ids.end(), m_t5_cache_ids.begin())
        && std::equal(m_attention_mask.begin(), m_attention_mask.end(), m_t5_cache_mask.begin());
    const std::vector<float>* precomputed_t5 = reuse_t5 ? &m_cached_t5_masked : nullptr;

    if (reuse_t5)
    {
        DBG("InferenceWorker: job #" + juce::String(job.job_id) + " T5 cache hit (reusing masked embeddings)");
        streamgen_log("InferenceWorker::process_job id=" + juce::String(job.job_id) + " t5_cache=hit");
    }
    else
    {
        DBG("InferenceWorker: job #" + juce::String(job.job_id) + " T5 cache miss (will run ONNX T5)");
        streamgen_log("InferenceWorker::process_job id=" + juce::String(job.job_id) + " t5_cache=miss");
    }

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
        job.cfg_scale,
        nullptr,
        {},
        "",
        precomputed_t5
    );

    if (!reuse_t5)
    {
        m_cached_t5_masked = m_pipeline->last_masked_t5_embeddings();
        m_t5_cache_ids = m_input_ids;
        m_t5_cache_mask = m_attention_mask;
        m_have_t5_cache = true;
    }

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
    streamgen_log("InferenceWorker::write_output id=" + juce::String(job.job_id)
        + " start=" + juce::String(job.output_start_sample())
        + " gen_samples=" + juce::String(static_cast<int>(gen_samples))
        + " xfade=" + juce::String(xfade)
        + " wall_ms=" + juce::String(total_ms, 1)
        + " abs_pos_now=" + juce::String(m_processor.scheduler().absolute_sample_pos()));

    if (m_timeline != nullptr)
        m_timeline->record_completed(job.job_id, job, total_ms, gen_samples);

    // Update timing + full snapshot for operator dashboard
    const auto& pipeline_timing = m_pipeline->timing();
    const auto& diag = m_pipeline->diagnostics();
    uint64_t attn_positions = 0;
    for (int64_t m : m_attention_mask)
    {
        if (m != 0)
            ++attn_positions;
    }

    {
        std::lock_guard<std::mutex> lock(m_worker_state_mutex);
        m_last_timing.vae_encode_ms = pipeline_timing.vae_encode_ms;
        m_last_timing.t5_encode_ms = pipeline_timing.t5_encode_ms;
        m_last_timing.sampling_total_ms = pipeline_timing.sampling_total_ms;
        m_last_timing.vae_decode_ms = pipeline_timing.vae_decode_ms;
        m_last_timing.total_ms = total_ms;
        m_last_timing.steps = job.steps;

        m_last_snapshot.timing = pipeline_timing;
        m_last_snapshot.diagnostics = diag;
        m_last_snapshot.job = job;
        m_last_snapshot.t5_sequence_length = m_input_ids.size();
        m_last_snapshot.t5_attention_nonzero_positions = attn_positions;
        m_last_snapshot.wall_clock_ms = total_ms;
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
