add_library(Matcher
    keypointmatcher.cpp
    keypointmatcher.h
    )

target_link_libraries(Matcher
    PUBLIC Core)

target_include_directories(Matcher
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
