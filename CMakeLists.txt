cmake_minimum_required(VERSION 2.8)
project(regopt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

# set make's verbosity
set(CMAKE_VERBOSE_MAKEFILE OFF) 

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(BUILD_SHARED_LIBS true)


# definitions
add_definitions("-DPROJECT_DATA_DIR=${CMAKE_SOURCE_DIR}/data")


# find boost
find_package(Boost COMPONENTS system python thread program_options REQUIRED)

# find openrave for includes in sco
find_package(OpenRAVE REQUIRED)
include_directories(${OpenRAVE_INCLUDE_DIRS})


# find eigen
find_package(Eigen REQUIRED)
add_definitions("-DEIGEN_DEFAULT_TO_ROW_MAJOR")
include_directories(${EIGEN_INCLUDE_DIRS})



# John's sco stuff
include_directories($ENV{TRAJOPT_SOURCE_DIR}/src
                    $ENV{TRAJOPT_SOURCE_DIR}/utils)
link_directories ($ENV{TRAJOPT_BUILD_DIR}/lib)


## print all cmake variables
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


add_subdirectory(lib)
add_subdirectory(src)
