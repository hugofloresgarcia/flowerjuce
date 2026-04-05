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
    add_subdirectory(sao_cli)
    add_subdirectory(zenon_cli)
    add_subdirectory(streamgen)
    add_subdirectory(streamgen_cli)
    add_subdirectory(wham)
endif()

