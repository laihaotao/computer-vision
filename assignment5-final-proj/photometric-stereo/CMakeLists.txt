cmake_minimum_required(VERSION 3.6)
project(photometric_stereo)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(photometric_stereo ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(photometric_stereo ${OpenCV_LIBS})