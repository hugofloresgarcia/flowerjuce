# flowerjuce shared library

add_library(flowerjuce STATIC)

# Engine source files
target_sources(flowerjuce PRIVATE
    LooperEngine/LooperTrackEngine.cpp
    LooperEngine/TapeLoop.cpp
    LooperEngine/LooperWriteHead.cpp
    LooperEngine/LooperReadHead.cpp
    LayerCakeEngine/LayerCakeEngine.cpp
    LayerCakeEngine/GrainVoice.cpp
    LayerCakeEngine/LayerCakeEnvelope.cpp
    LayerCakeEngine/Metro.cpp
)

# Engine headers
target_sources(flowerjuce PRIVATE
    LooperEngine/LooperTrackEngine.h
    LooperEngine/MultiTrackLooperEngine.h
    LooperEngine/TapeLoop.h
    LooperEngine/LooperWriteHead.h
    LooperEngine/LooperReadHead.h
    LooperEngine/OutputBus.h
    LayerCakeEngine/LayerCakeEngine.h
    LayerCakeEngine/LayerCakeTypes.h
    LayerCakeEngine/GrainVoice.h
    LayerCakeEngine/LayerCakeEnvelope.h
    LayerCakeEngine/Metro.h
)

# DSP source files
target_sources(flowerjuce PRIVATE
    DSP/OnsetDetector.cpp
    DSP/LowPassFilter.cpp
    DSP/MultiChannelLoudnessMeter.cpp
    DSP/PeakMeter.cpp
    DSP/KnobSweepRecorder.cpp
    DSP/LfoUGen.cpp
)

# DSP headers
target_sources(flowerjuce PRIVATE
    DSP/OnsetDetector.h
    DSP/LowPassFilter.h
    DSP/MultiChannelLoudnessMeter.h
    DSP/PeakMeter.h
    DSP/KnobSweepRecorder.h
    DSP/LfoUGen.h
)

# Utils source files (removed - Utils directory doesn't exist)
# target_sources(flowerjuce PRIVATE
#     Utils/Utils.cpp
# )

# Utils headers (removed - Utils directory doesn't exist)
# target_sources(flowerjuce PRIVATE
#     Utils/Utils.h
# )

# GradioClient source files
target_sources(flowerjuce PRIVATE
    GradioClient/GradioClient.cpp
)

# GradioClient headers
target_sources(flowerjuce PRIVATE
    GradioClient/GradioClient.h
)

# Panners source files
target_sources(flowerjuce PRIVATE
    Panners/PanningUtils.cpp
    Panners/StereoPanner.cpp
    Panners/QuadPanner.cpp
    Panners/CLEATPanner.cpp
    Panners/Panner2DComponent.cpp
    Panners/PathGeneratorButtons.cpp
)

# Panners headers
target_sources(flowerjuce PRIVATE
    Panners/Panner.h
    Panners/PanningUtils.h
    Panners/StereoPanner.h
    Panners/QuadPanner.h
    Panners/CLEATPanner.h
    Panners/Panner2DComponent.h
    Panners/PathGeneratorButtons.h
)

# Components source files (renamed from Shared)
target_sources(flowerjuce PRIVATE
    Components/WaveformDisplay.cpp
    Components/DualWaveformDisplay.cpp
    Components/TransportControls.cpp
    Components/ParameterKnobs.cpp
    Components/LevelControl.cpp
    Components/InputSelector.cpp
    Components/GradioUtilities.cpp
    Components/MidiLearnManager.cpp
    Components/ConfigManager.cpp
    Components/VariationSelector.cpp
    Components/SinksWindow.cpp
    Components/MultiChannelMeter.cpp
)

# Components headers
target_sources(flowerjuce PRIVATE
    Components/WaveformDisplay.h
    Components/DualWaveformDisplay.h
    Components/TransportControls.h
    Components/ParameterKnobs.h
    Components/LevelControl.h
    Components/InputSelector.h
    Components/GradioUtilities.h
    Components/MidiLearnManager.h
    Components/MidiLearnComponent.h
    Components/ConfigManager.h
    Components/VariationSelector.h
    Components/SettingsDialog.h
    Components/ModelParameterDialog.h
    Components/AudioInfoDisplay.h
    Components/SinksWindow.h
    Components/MultiChannelMeter.h
)

# Sync source files
target_sources(flowerjuce PRIVATE
    Sync/LinkSyncStrategy.cpp
    Sync/InternalSyncStrategy.cpp
)

# Sync headers
target_sources(flowerjuce PRIVATE
    Sync/SyncInterface.h
    Sync/LinkSyncStrategy.h
    Sync/InternalSyncStrategy.h
)

# Debug headers
target_sources(flowerjuce PRIVATE
    Debug/DebugAudioRate.h
)

# CustomLookAndFeel header
target_sources(flowerjuce PRIVATE
    CustomLookAndFeel.h
)

# ClickSynth and Sampler (removed - files don't exist, needed by VampNetTrackEngine which also doesn't exist)
# target_sources(flowerjuce PRIVATE
#     ClickSynth.cpp
#     ClickSynth.h
#     Sampler.cpp
#     Sampler.h
# )

# Set include directories - this allows includes like <flowerjuce/LooperEngine/...>
# From libs/flowerjuce/, we need to go up to root, then include libs/ so flowerjuce/ resolves correctly
get_filename_component(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
target_include_directories(flowerjuce PUBLIC
    ${PROJECT_ROOT}/libs
)

if(NOT TARGET Ableton::Link)
    include(${PROJECT_ROOT}/third_party/link/AbletonLinkConfig.cmake)
endif()

if(IOS)
    target_compile_definitions(flowerjuce PUBLIC LINK_PLATFORM_IOS=1)
endif()

# On Linux, juce_core uses libcurl but the module metadata doesn't declare it
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR CMAKE_SYSTEM_NAME MATCHES ".*BSD")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(LIBCURL REQUIRED libcurl)
    target_include_directories(flowerjuce PUBLIC ${LIBCURL_INCLUDE_DIRS})
    target_link_libraries(flowerjuce PUBLIC ${LIBCURL_LIBRARIES})
endif()

# Link JUCE modules
target_link_libraries(flowerjuce PUBLIC
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_dsp
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
    Ableton::Link
)

