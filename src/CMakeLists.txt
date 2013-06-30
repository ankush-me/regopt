include_directories(
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/src
)

add_executable(reg problem_description.cpp fastrapp.cpp)
target_link_libraries(reg sco)
