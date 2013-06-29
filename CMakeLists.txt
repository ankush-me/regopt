cmake_minimum_required(VERSION 2.6)
project(regopt)

find_package(Boost COMPONENTS system REQUIRED)

find_package(Eigen REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})

add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
add_definitions("-DPROJECT_DATA_DIR=${CMAKE_SOURCE_DIR}/data")

add_subdirectory(lib)
add_subdirectory(src)
