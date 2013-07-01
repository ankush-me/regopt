include_directories(
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(utils)

add_executable(reg sqp_reg.cpp problem_description.cpp fastrapp.cpp)
target_link_libraries(reg sco reg_utils)
