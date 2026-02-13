#pragma once
#include <juce_core/juce_core.h>
#include <juce_audio_devices/juce_audio_devices.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <atomic>
#include <memory>

namespace LatencyMeasurement {

enum class SignalQuality {
    Good,              // Valid measurement
    NoAudio,           // Recording too quiet
    NoLoop,            // No correlation peak found
    NoisyEnvironment,  // Signal masked by noise
    LowQuality         // Peak not sharp enough
};

struct LatencyResult {
    int latencySamples = 0;
    double latencyMs = 0.0;
    double sampleRate = 0.0;
    bool isValid = false;
    SignalQuality quality = SignalQuality::Good;
    juce::String warningMessage;
    
    // Signal quality metrics
    double recordingRMS = 0.0;
    double peakRatio = 0.0;      // peak / RMS of correlation
    double energyRatio = 0.0;    // sweep period energy / silence energy
    double peakSharpness = 0.0;  // peak / average of neighbors
    
    juce::String toString() const {
        juce::String result;
        if (isValid) {
            result << "Latency: " << latencySamples << " samples\n"
                   << "Latency: " << juce::String(latencyMs, 2) << " ms\n"
                   << "Sample Rate: " << juce::String(sampleRate, 0) << " Hz";
        }
        if (!isValid && warningMessage.isNotEmpty())
            result << warningMessage;
        return result;
    }
};

class LatencyMeasurementEngine : public juce::AudioIODeviceCallback {
public:
    LatencyMeasurementEngine();
    ~LatencyMeasurementEngine() override;
    
    bool startMeasurement();
    bool isMeasurementComplete() const { return measurementComplete.load(); }
    LatencyResult computeLatency();
    
    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;
    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
        int numInputChannels, float* const* outputChannelData, int numOutputChannels,
        int numSamples, const juce::AudioIODeviceCallbackContext& context) override;
    
private:
    double sampleRate = 48000.0;
    int sweepSamples = 0;  // Duration of sweep (excluding silence)
    std::vector<float> sweepSignal, recordedSignal;
    std::atomic<bool> testRunning{false}, measurementComplete{false};
    std::atomic<int> currentPosition{0};
    int totalSamples = 0;
    std::unique_ptr<juce::dsp::FFT> fft;
    static constexpr int fftOrder = 17;
    LatencyResult lastResult;
    
    struct CorrelationResult {
        int peakIndex = 0;
        float peakValue = 0.0f;
        float correlationRMS = 0.0f;
        float peakSharpness = 0.0f;
    };
    
    CorrelationResult findPeakCorrelation(const std::vector<float>& signal1, const std::vector<float>& signal2);
    double computeRMS(const std::vector<float>& signal, int start, int length);
    void validateSignalQuality(LatencyResult& result);
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LatencyMeasurementEngine)
};

}
