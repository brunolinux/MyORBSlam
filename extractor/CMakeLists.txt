add_library(Extractor
    abstractextractor.h
    orbextractor.cpp
    orbextractor.h
    pyramidlevelextractor.cpp
    pyramidlevelextractor.h
    quadtreenode.cpp
    quadtreenode.h)

target_link_libraries(Extractor
    PUBLIC ${OpenCV_LIBS})

target_include_directories(Extractor
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
