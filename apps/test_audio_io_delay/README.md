# Audio Latency Measurement App

A JUCE application for measuring audio system round-trip latency using logarithmic frequency sweep and cross-correlation.

## Overview

Measures the time for audio to travel from output → speakers/headphones → air → microphone → input. Uses a 1-second log sweep (100-10,000 Hz) and FFT cross-correlation to detect the time offset.

## Building

```bash
# From project root
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target LatencyMeasurementApp --config Release -j8
```

Built app location: `build/apps/test_audio_io_delay/LatencyMeasurementApp/LatencyMeasurementApp.app`

## Running

```bash
open build/apps/test_audio_io_delay/LatencyMeasurementApp/LatencyMeasurementApp.app
```

Or use the convenience script from project root:

```bash
./run_latency_app.sh
```

## Usage

1. **Setup**: Connect audio output to input (cable or acoustic loopback)
2. **Launch**: Select audio devices in startup dialog
3. **Measure**: Click "Start Test" and wait ~2 seconds
4. **Results**: Latency shown in samples, ms, and seconds

### Typical Values

- Built-in audio: 20-50 ms
- USB interface: 10-30 ms  
- Bluetooth: 100-300 ms

### Tips

- Use wired devices (not Bluetooth) for accurate results
- Keep environment quiet
- Position output close to microphone for acoustic loopback
- Run multiple tests and average

## Technical Details

- **Method**: Log sweep cross-correlation (Farina, 2000)
- **Framework**: JUCE (C++17)
- **DSP**: FFT order 17 (131k samples)
- **Thread-safe**: Real-time audio callback with atomic flags
