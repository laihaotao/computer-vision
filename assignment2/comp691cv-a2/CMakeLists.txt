cmake_minimum_required(VERSION 3.6)
project(comp691cv_a2)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(comp691cv_a2 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(comp691cv_a2 ${OpenCV_LIBS})