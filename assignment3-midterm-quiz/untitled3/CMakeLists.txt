cmake_minimum_required(VERSION 3.6)
project(untitled3)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp util.cpp HarrisDetector.cpp HarrisDetector.h util.h)
add_executable(untitled3 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(untitled3 ${OpenCV_LIBS})