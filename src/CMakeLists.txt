include_directories(
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(utils)
add_subdirectory(tests)

add_library(sqp_reg sqpreg.cpp problem_description.cpp fastrapp.cpp)

# make the boost python module library
boost_python_module(sqpreg sqpreg.cpp)