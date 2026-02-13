#include "LatencyMeasurementEngine.h"
#include "SweepGenerator.h"
#include <algorithm>

namespace LatencyMeasurement
{

    LatencyMeasurementEngine::LatencyMeasurementEngine() {}
    LatencyMeasurementEngine::~LatencyMeasurementEngine() { testRunning = false; }

    bool LatencyMeasurementEngine::startMeasurement()
    {
        if (testRunning.load())
            return false;

        measurementComplete = false;
        currentPosition = 0;
        sweepSignal = SweepGenerator::generateDefaultSweep(sampleRate);
        totalSamples = static_cast<int>(sweepSignal.size());
        sweepSamples = static_cast<int>(sampleRate);
        recordedSignal.assign(totalSamples, 0.0f);
        lastResult = LatencyResult();
        testRunning = true;

        return true;
    }

    void LatencyMeasurementEngine::audioDeviceAboutToStart(juce::AudioIODevice *device)
    {
        if (device)
            sampleRate = device->getCurrentSampleRate();
    }

    void LatencyMeasurementEngine::audioDeviceStopped() { testRunning = false; }

    void LatencyMeasurementEngine::audioDeviceIOCallbackWithContext(
        const float *const *inputChannelData, int numInputChannels,
        float *const *outputChannelData, int numOutputChannels,
        int numSamples, const juce::AudioIODeviceCallbackContext &)
    {
        for (int ch = 0; ch < numOutputChannels; ++ch)
            if (outputChannelData[ch])
                std::fill_n(outputChannelData[ch], numSamples, 0.0f);

        if (!testRunning.load())
            return;

        int pos = currentPosition.load();
        if (pos >= totalSamples)
        {
            testRunning = false;
            measurementComplete = true;
            return;
        }

        int samplesToProcess = std::min(numSamples, totalSamples - pos);

        if (numOutputChannels > 0 && outputChannelData[0])
        {
            for (int i = 0; i < samplesToProcess; ++i)
                outputChannelData[0][i] = sweepSignal[pos + i];
            if (numOutputChannels > 1 && outputChannelData[1])
                std::copy_n(outputChannelData[0], samplesToProcess, outputChannelData[1]);
        }

        if (numInputChannels > 0 && inputChannelData[0])
            for (int i = 0; i < samplesToProcess; ++i)
                recordedSignal[pos + i] = inputChannelData[0][i];

        currentPosition.store(pos + samplesToProcess);
        if (pos + samplesToProcess >= totalSamples)
        {
            testRunning = false;
            measurementComplete = true;
        }
    }

    LatencyResult LatencyMeasurementEngine::computeLatency()
    {
        if (!measurementComplete.load())
            return lastResult;

        LatencyResult result;
        result.sampleRate = sampleRate;

        // Compute signal quality metrics
        result.recordingRMS = computeRMS(recordedSignal, 0, totalSamples);

        // Energy during sweep vs silence
        double sweepEnergy = computeRMS(recordedSignal, 0, sweepSamples);
        double silenceEnergy = computeRMS(recordedSignal, sweepSamples, totalSamples - sweepSamples);
        result.energyRatio = (silenceEnergy > 1e-10) ? (sweepEnergy / silenceEnergy) : 100.0;

        // Find correlation peak
        auto corrResult = findPeakCorrelation(recordedSignal, sweepSignal);
        result.latencySamples = corrResult.peakIndex;
        result.latencyMs = (result.latencySamples / sampleRate) * 1000.0;
        result.peakRatio = (corrResult.correlationRMS > 1e-10) ? (corrResult.peakValue / corrResult.correlationRMS) : 0.0;
        result.peakSharpness = corrResult.peakSharpness;

        // Validate signal quality
        validateSignalQuality(result);

        lastResult = result;
        return result;
    }

    double LatencyMeasurementEngine::computeRMS(const std::vector<float> &signal, int start, int length)
    {
        if (length <= 0 || start < 0 || start + length > static_cast<int>(signal.size()))
            return 0.0;

        double sum = 0.0;
        for (int i = start; i < start + length; ++i)
            sum += signal[i] * signal[i];
        return std::sqrt(sum / length);
    }

    void LatencyMeasurementEngine::validateSignalQuality(LatencyResult &result)
    {
        // Thresholds (may need tuning)
        constexpr double minRecordingRMS = 0.001; // Minimum recording level
        constexpr double minPeakRatio = 8.0;      // Peak must be 8x above RMS
        constexpr double minEnergyRatio = 1.5;    // Sweep energy 1.5x silence
        constexpr double minPeakSharpness = 1.5;  // Peak must be 1.5x neighbors

        // Check 1: Recording too quiet (no audio or very quiet environment)
        if (result.recordingRMS < minRecordingRMS)
        {
            result.isValid = false;
            result.quality = SignalQuality::NoAudio;
            result.warningMessage = "No audio detected.\n\n"
                                    "Possible causes:\n"
                                    "- Microphone not working or muted\n"
                                    "- Quiet environment with no feedback loop\n\n"
                                    "Please check microphone and place speakers close to it.";
            return;
        }

        // Check 2: No clear correlation peak (no loop detected)
        if (result.peakRatio < minPeakRatio)
        {
            result.isValid = false;
            result.quality = SignalQuality::NoLoop;
            result.warningMessage = "No feedback loop detected.\n\n"
                                    "The sweep signal was not captured by the microphone.\n"
                                    "Please place speakers/headphones closer to the microphone.";
            return;
        }

        // Check 3: Signal masked by noise (energy ratio too low)
        if (result.energyRatio < minEnergyRatio)
        {
            result.isValid = false;
            result.quality = SignalQuality::NoisyEnvironment;
            result.warningMessage = "Signal masked by environmental noise.\n\n"
                                    "The recording has too much background noise.\n"
                                    "Please test in a quieter environment or increase volume.";
            return;
        }

        // Check 4: Peak not sharp enough (low quality)
        if (result.peakSharpness < minPeakSharpness)
        {
            result.isValid = false;
            result.quality = SignalQuality::LowQuality;
            result.warningMessage = "Low signal quality - result may be inaccurate.\n\n"
                                    "Peak sharpness: " +
                                    juce::String(result.peakSharpness, 2) + " (need >= " + juce::String(minPeakSharpness, 1) + ")\n"
                                                                                                                               "Estimated latency: " +
                                    juce::String(result.latencyMs, 2) + " ms (" + juce::String(result.latencySamples) + " samples)\n\n"
                                                                                                                        "The correlation peak is not clear enough.\n"
                                                                                                                        "Please reduce background noise or increase speaker volume.";
            return;
        }

        // Check 5: Latency in reasonable range
        if (result.latencyMs < 1.0 || result.latencyMs > 1000.0)
        {
            result.isValid = false;
            result.quality = SignalQuality::NoLoop;
            result.warningMessage = "Measured latency out of expected range.\n\n"
                                    "Please check audio connections and try again.";
            return;
        }

        // All checks passed
        result.isValid = true;
        result.quality = SignalQuality::Good;
    }

    LatencyMeasurementEngine::CorrelationResult LatencyMeasurementEngine::findPeakCorrelation(
        const std::vector<float> &signal1, const std::vector<float> &signal2)
    {
        CorrelationResult result;

        if (!fft)
            fft = std::make_unique<juce::dsp::FFT>(fftOrder);
        int fftSize = 1 << fftOrder;

        std::vector<juce::dsp::Complex<float>> complex1(fftSize, {0.0f, 0.0f});
        std::vector<juce::dsp::Complex<float>> complex2(fftSize, {0.0f, 0.0f});

        size_t copySize = std::min(signal1.size(), static_cast<size_t>(fftSize));
        for (size_t i = 0; i < copySize; ++i)
        {
            complex1[i] = {signal1[i], 0.0f};
            complex2[i] = {signal2[i], 0.0f};
        }

        fft->perform(complex1.data(), complex1.data(), false);
        fft->perform(complex2.data(), complex2.data(), false);

        // Cross-correlation: signal1 * conj(signal2)
        std::vector<juce::dsp::Complex<float>> cross(fftSize);
        for (int i = 0; i < fftSize; ++i)
        {
            float r1 = complex1[i].real(), i1 = complex1[i].imag();
            float r2 = complex2[i].real(), i2 = complex2[i].imag();
            cross[i] = {r1 * r2 + i1 * i2, i1 * r2 - r1 * i2};
        }

        fft->perform(cross.data(), cross.data(), true);

        // Find peak and compute RMS of correlation
        int searchRange = std::min(static_cast<int>(signal2.size()), fftSize);
        double sumSq = 0.0;

        for (int i = 0; i < searchRange; ++i)
        {
            float value = std::abs(cross[i].real());
            sumSq += value * value;
            if (value > result.peakValue)
            {
                result.peakValue = value;
                result.peakIndex = i;
            }
        }

        result.correlationRMS = static_cast<float>(std::sqrt(sumSq / searchRange));

        // Compute peak sharpness: peak / average of neighbors (±50 samples)
        constexpr int neighborRange = 50;
        int startIdx = std::max(0, result.peakIndex - neighborRange);
        int endIdx = std::min(searchRange, result.peakIndex + neighborRange + 1);
        float neighborSum = 0.0f;
        int neighborCount = 0;

        for (int i = startIdx; i < endIdx; ++i)
        {
            if (i != result.peakIndex)
            {
                neighborSum += std::abs(cross[i].real());
                neighborCount++;
            }
        }

        float neighborAvg = (neighborCount > 0) ? (neighborSum / neighborCount) : 1.0f;
        result.peakSharpness = (neighborAvg > 1e-10f) ? (result.peakValue / neighborAvg) : 0.0f;

        return result;
    }

}
