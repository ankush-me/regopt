cmake_minimum_required(VERSION 2.8)
project(regopt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")
set(CMAKE_VERBOSE_MAKEFILE OFF) 

# definitions
add_definitions("-DPROJECT_DATA_DIR=${CMAKE_SOURCE_DIR}/data")

find_package(Boost COMPONENTS system REQUIRED)

find_package(Eigen REQUIRED)
add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
include_directories(${EIGEN_INCLUDE_DIRS})

# John's sco stuff
include_directories($ENV{TRAJOPT_SOURCE_DIR}/src
                    $ENV{TRAJOPT_SOURCE_DIR}/utils)
link_directories ($ENV{TRAJOPT_BINARY_DIR}/lib) 


add_subdirectory(lib)
add_subdirectory(src)
