add_library(Camera
    abstractcamera.cpp
    abstractcamera.h
    monocamera.cpp
    monocamera.h
    rgbdcamera.cpp
    rgbdcamera.h
    stereocamera.cpp
    stereocamera.h
    )

target_link_libraries(Camera
    PUBLIC ${OpenCV_LIBS}
    PRIVATE Extractor)

target_include_directories(Camera
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
