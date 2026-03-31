#include "DiagnosticLogWriter.h"
#include "InferenceWorker.h"

#include <iomanip>
#include <sstream>

namespace streamgen {

DiagnosticLogWriter::DiagnosticLogWriter(
    StreamGenProcessor& processor,
    std::function<InferenceWorker*()> get_worker,
    const juce::File& log_file)
    : juce::Thread("StreamGen DiagnosticLog"),
      m_processor(processor),
      m_get_worker(std::move(get_worker)),
      m_log_file(log_file)
{
}

DiagnosticLogWriter::~DiagnosticLogWriter()
{
    stopThread(3000);
}

void DiagnosticLogWriter::run()
{
    m_stream = std::make_unique<juce::FileOutputStream>(m_log_file);
    if (m_stream == nullptr || !m_stream->openedOk())
        return;

    const juce::String header =
        "time_unix_ms,audio_cb_count,ema_in_rms,ema_out_rms,worker_busy,gen_count,last_job_id,wall_ms_last\n";
    m_stream->writeText(header, false, false, nullptr);

    while (!threadShouldExit())
    {
        wait(200);

        AudioThreadTelemetry::Snapshot audio;
        m_processor.audio_telemetry().copy_snapshot(audio);

        const bool busy = m_processor.scheduler().status.worker_busy.load(std::memory_order_relaxed);
        const int gen_count = static_cast<int>(
            m_processor.scheduler().status.generation_count.load(std::memory_order_relaxed));
        const int64_t job_id = m_processor.scheduler().status.last_job_id.load(std::memory_order_relaxed);

        double wall_ms = 0.0;
        InferenceWorker* w = m_get_worker();
        if (w != nullptr && w->is_loaded())
            wall_ms = w->last_snapshot().wall_clock_ms;

        const juce::int64 now_ms = juce::Time::getCurrentTime().toMilliseconds();

        std::ostringstream line;
        line << now_ms << ","
             << audio.callback_count << ","
             << std::fixed << std::setprecision(6) << audio.ema_input_rms << ","
             << audio.ema_output_rms << ","
             << (busy ? 1 : 0) << ","
             << gen_count << ","
             << job_id << ","
             << std::setprecision(3) << wall_ms << "\n";

        m_stream->writeText(juce::String(line.str()), false, false, nullptr);
        m_stream->flush();
    }

    m_stream.reset();
}

} // namespace streamgen
