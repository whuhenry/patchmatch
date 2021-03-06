cmake_minimum_required(VERSION 3.5)
project(patchmatch)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

SET(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.56 REQUIRED COMPONENTS log)

if(MSVC)
  # Force to always compile with W4
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W1" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
  endif()
endif()

add_executable(patchmatch main.cpp Image.cpp Image.h PatchMatchAlg.cpp PatchMatchAlg.h)
target_link_libraries(patchmatch ${OpenCV_LIBS} Boost::log)