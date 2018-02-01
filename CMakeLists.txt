cmake_minimum_required(VERSION 3.9)
project(patchmatch)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

add_executable(patchmatch main.cpp Image.cpp Image.h)
target_link_libraries(patchmatch ${OpenCV_LIBS})