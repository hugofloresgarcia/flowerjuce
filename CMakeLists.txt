cmake_minimum_required(VERSION 3.25)

# Old FetchContent deps (e.g. cnpy) ship CMakeLists with cmake_minimum_required(2.8); allow them
# under newer CMake toolchains.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.31)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
endif()

# Version is defined here as the single source of truth
# This version is used by:
# - All CMake targets (via ${PROJECT_VERSION})
# - Build scripts (extracted via grep/sed)
# - GitHub releases (via build script)
project(TapeLooper VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_definitions(JUCE_MODAL_LOOPS_PERMITTED)

# Add JUCE
add_subdirectory(JUCE)

# Add libraries
add_subdirectory(third_party)
add_subdirectory(libs)

# Add apps
add_subdirectory(apps)

# Add tests (skip for iOS builds)
if(NOT IOS)
    add_subdirectory(tests)
    
    # Build AudioPluginHost for testing plugins
    add_subdirectory(JUCE/extras/AudioPluginHost)
endif()

# Add plugins
# add_subdirectory(plugins)
