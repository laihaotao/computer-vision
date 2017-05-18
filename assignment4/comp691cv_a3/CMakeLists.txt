cmake_minimum_required(VERSION 3.6)
project(comp691cv_a3)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp util.cpp HarrisDetector.cpp stitch_util.cpp stitch_util.h HarrisDetector.h util.h)
add_executable(comp691cv_a3 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(comp691cv_a3 ${OpenCV_LIBS})