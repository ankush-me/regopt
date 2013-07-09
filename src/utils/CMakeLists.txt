# utility code

add_library(reg_utils eigen_utils.cpp opt_utils.cpp colorize.cpp)
target_link_libraries(reg_utils sco)
