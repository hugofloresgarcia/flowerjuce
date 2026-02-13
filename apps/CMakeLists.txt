# Apps directory - includes all app subdirectories

# For iOS builds, only build LayerCake (iPad-only app)
if(IOS)
    add_subdirectory(layercake)
else()
    add_subdirectory(basic)
    # add_subdirectory(text2sound)  # Directory does not exist
    add_subdirectory(text2sound4all)
    add_subdirectory(embeddingsampler)
    add_subdirectory(layercake)
    add_subdirectory(test_audio_io_delay)
endif()

