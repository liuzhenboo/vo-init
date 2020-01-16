cmake_minimum_required(VERSION 2.8)
project(ORB-MVO)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Check C++11 or C++0x support
add_definitions(-DCOMPILEDWITHC11)
if( CMAKE_BUILD_TYPE MATCHES "Debug" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -g -march=native -Wno-deprecated-declarations")
else( CMAKE_BUILD_TYPE MATCHES "Release" )
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -msse3 -std=c++11 -pthread -O3 -march=native -Wno-deprecated-declarations")
endif( CMAKE_BUILD_TYPE MATCHES "Debug" )

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

#find_package(OpenCV 2.4.10 REQUIRED)
find_package(OpenCV)

find_package(Eigen3 REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Tracking.cpp
src/ORBextractor.cpp
src/ORBmatcher.cpp
src/Frame.cpp
src/Initializer.cpp
src/Converter.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so

)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/Examples/Monocular)
add_executable(mono_tum Examples/Monocular/mono_tum.cpp)
target_link_libraries(mono_tum ${PROJECT_NAME})




