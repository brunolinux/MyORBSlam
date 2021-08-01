cmake_minimum_required(VERSION 3.10)
#cmake_policy(SET CMP0110 NEW)

project(my_orb_slam LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Catch2 REQUIRED)

add_subdirectory(camera)
add_subdirectory(extractor)
add_subdirectory(solver)
add_subdirectory(test)

add_executable(my_orb_slam main.cpp)
