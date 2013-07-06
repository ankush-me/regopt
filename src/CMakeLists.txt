include_directories(
  ${CMAKE_SOURCE_DIR}/lib
  ${CMAKE_SOURCE_DIR}/src
)

add_subdirectory(utils)
add_subdirectory(tests)

add_library(sqp_reg sqpreg.cpp
                    tps_reg_fit_problem.cpp
                    tps_fit_problem.cpp
                    rotreg.cpp
                    tps_costs.cpp)

add_subdirectory(sqpregpy)