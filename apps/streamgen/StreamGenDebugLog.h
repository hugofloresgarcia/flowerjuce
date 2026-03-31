#pragma once

#include <atomic>
#include <iostream>

#include <juce_core/juce_core.h>

namespace streamgen {

/// Writes to stderr so logs appear in Terminal and Release builds (unlike DBG()).
inline std::atomic<bool> g_streamgen_stderr_trace{true};

/// Every audio callback emits one line (can be ~50–200/s). Default false: first 4 lines then every 256th block (~few/sec at 512-sample buffers).
inline std::atomic<bool> g_streamgen_log_every_audio_block{false};

inline void streamgen_log(const juce::String& line)
{
    if (!g_streamgen_stderr_trace.load(std::memory_order_relaxed))
        return;
    const juce::String prefixed = "[StreamGen] " + line;
    // Single sink: FileLogger::writeToLog also ends up on stderr in common JUCE setups; do not duplicate with std::cerr.
    juce::Logger::writeToLog(prefixed);
}

/// For very hot paths (e.g. audio); returns whether this invocation should print.
inline bool streamgen_log_audio_throttle(int& call_counter)
{
    if (g_streamgen_log_every_audio_block.load(std::memory_order_relaxed))
        return g_streamgen_stderr_trace.load(std::memory_order_relaxed);
    if (!g_streamgen_stderr_trace.load(std::memory_order_relaxed))
        return false;
    ++call_counter;
    return call_counter <= 4 || (call_counter % 256 == 0);
}

} // namespace streamgen
