cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.56 REQUIRED COMPONENTS log program_options)

find_package(CUDA REQUIRED)

cuda_add_executable(patchmatch Image.cpp Image.h PatchMatchAlg.cpp PatchMatchAlg.h
               patchmatch_gpu.cu patchmatch_gpu.h helper_cuda.h helper_string.h helper_timer.h
               exception.h main.cpp)
target_link_libraries(patchmatch ${OpenCV_LIBS} Boost::log Boost::program_options)