find_package(Threads REQUIRED)


add_library(Initializer
    abstractinitializer.h
    monoinitializer.cpp
    monoinitializer.h
    )

target_link_libraries(Initializer
    PUBLIC ${OpenCV_LIBS} Matcher
    )

target_link_libraries(Initializer
    PRIVATE ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    PRIVATE Threads::Threads
    )

target_include_directories(Initializer
    PRIVATE ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/DUtils)

target_include_directories(Initializer
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
