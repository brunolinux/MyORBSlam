add_library(Initializer
    abstractinitializer.h
    monoinitializer.cpp
    monoinitializer.h
    )

target_link_libraries(Initializer
    PUBLIC ${OpenCV_LIBS}
    PRIVATE Core)

target_include_directories(Initializer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
