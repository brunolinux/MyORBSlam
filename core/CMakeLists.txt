add_library(Core
    frame.cpp
    frame.h)

target_link_libraries(Core
    PUBLIC ${OpenCV_LIBS} Extractor
    )

target_include_directories(Core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
