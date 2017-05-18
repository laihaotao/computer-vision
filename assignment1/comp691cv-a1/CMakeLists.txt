cmake_minimum_required(VERSION 3.6)
project(comp691cv-a1)

set(CMAKE_CXX_STANDARD 98)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp)
add_executable(comp691cv-a1 ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(comp691cv-a1 ${OpenCV_LIBS})